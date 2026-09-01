"""Official-style cumulative scribbles from a champion iter0 error mask."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

import cc3d
import numpy as np
from scipy.ndimage import find_objects
from scipy.spatial import ConvexHull, QhullError


STRATEGIES = ("centerline", "random", "boundary")


@lru_cache(maxsize=1)
def _official_module():
    source = os.environ.get("AUTOPET_OFFICIAL_SCRIBBLE_SIM")
    if not source:
        raise RuntimeError("AUTOPET_OFFICIAL_SCRIBBLE_SIM must point to official simulate_scribbles.py")
    spec = importlib.util.spec_from_file_location("autopetv_official_simulate_scribbles", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official scribble simulator from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_simulator():
    return _official_module().simulate_scribble_from_label


def _simulate(mask: np.ndarray, strategy: str, seed: int):
    """Normalize the official helper's empty-mask two-value return."""
    if not np.any(mask):
        return [], False, 0
    result = _official_simulator()(mask.astype(np.uint8, copy=False), strategy=strategy, seed=seed)
    if len(result) == 2:
        coordinates, label_class = result
        return coordinates, label_class, len(coordinates)
    coordinates, label_class, size = result
    return coordinates, label_class, int(size)


def _correct_selected_component(working_prediction: np.ndarray, error: np.ndarray,
                                coordinates: list[list[int]], value: bool):
    """Assume the prompted 3-D error component is corrected before the next turn."""
    if not coordinates:
        return
    labels = cc3d.connected_components(error.astype(np.uint8, copy=False), connectivity=26)
    coordinate = tuple(int(item) for item in coordinates[0])
    component_id = int(labels[coordinate])
    if component_id == 0:
        raise RuntimeError(f"official scribble coordinate {coordinate} is outside its source error mask")
    working_prediction[labels == component_id] = value


def _prepare_error_components(error: np.ndarray):
    """Cache the official best 2-D slice for every 3-D error component.

    The official simulator repeatedly scans every z slice and selects the
    largest 2-D component. Correcting one prompt removes its entire 3-D
    component, so the candidates for all other components are invariant and
    can be reused on later turns.
    """
    labels_3d = cc3d.connected_components(error.astype(np.uint8, copy=False), connectivity=26)
    candidates = {}
    for z in range(error.shape[2]):
        slice_mask = error[:, :, z]
        if not np.any(slice_mask):
            continue
        labels_2d = cc3d.connected_components(slice_mask.astype(np.uint8, copy=False), connectivity=8)
        counts = np.bincount(labels_2d.ravel())
        bounding_boxes = find_objects(labels_2d)
        for component_2d, bounding_box in enumerate(bounding_boxes, start=1):
            if bounding_box is None:
                continue
            component_crop = labels_2d[bounding_box] == component_2d
            voxel_local = np.argwhere(component_crop)[0]
            row_offset = int(bounding_box[0].start)
            column_offset = int(bounding_box[1].start)
            component_3d = int(labels_3d[
                row_offset + int(voxel_local[0]),
                column_offset + int(voxel_local[1]),
                z,
            ])
            if component_3d == 0:
                raise RuntimeError("2-D error component is not represented in 3-D labels")
            previous = candidates.get(component_3d)
            rank = (z, component_2d)
            area = int(counts[component_2d])
            if previous is None or int(area) > previous["area"]:
                row_pad_before = int(row_offset > 0)
                row_pad_after = int(bounding_box[0].stop < labels_2d.shape[0])
                column_pad_before = int(column_offset > 0)
                column_pad_after = int(bounding_box[1].stop < labels_2d.shape[1])
                candidates[component_3d] = {
                    "area": area,
                    "rank": rank,
                    "z": z,
                    # Preserve the zero-valued neighborhood seen by the
                    # official boundary detector while avoiding a full-slice
                    # allocation for every small component.
                    "mask": np.pad(
                        component_crop,
                        (
                            (row_pad_before, row_pad_after),
                            (column_pad_before, column_pad_after),
                        ),
                    ),
                    "offset": (
                        row_offset - row_pad_before,
                        column_offset - column_pad_before,
                    ),
                }
    return labels_3d, candidates


def _best_component_scribble(candidates, active_components, strategy: str, seed: int):
    if not active_components:
        return [], 0, None
    component_id = min(
        active_components,
        key=lambda item: (-candidates[item]["area"], candidates[item]["rank"]),
    )
    candidate = candidates[component_id]
    module = _official_module()
    try:
        if strategy == "centerline":
            scribble, _ = _scribble_centerline_exact_fast(candidate["mask"])
        elif strategy == "boundary":
            scribble, _ = module.scribble_boundary(candidate["mask"], seed)
        else:
            scribble, _ = module.scribble_random(candidate["mask"], seed)
    except Exception:
        scribble, _ = module.scribble_random(candidate["mask"], seed)
    coordinates_2d = np.argwhere(scribble > 0)
    coordinates = [
        [
            int(coordinate[0]) + candidate["offset"][0],
            int(coordinate[1]) + candidate["offset"][1],
            int(candidate["z"]),
        ]
        for coordinate in coordinates_2d
    ]
    return coordinates, int(np.count_nonzero(scribble)), component_id


def _scribble_centerline_exact_fast(slice_mask, trunc_fraction=0.1):
    """Exact official centerline with an O(h^2) diameter on hull points.

    The official implementation materializes ``cdist(coords, coords)`` for all
    skeleton pixels. The Euclidean diameter must lie on the convex hull, so
    evaluating only hull points preserves the selected endpoints and path while
    avoiding pathological quadratic work on large skeletons.
    """
    module = _official_module()
    skeleton = module.skeletonize(slice_mask).astype(np.uint8)
    skeleton_components = cc3d.connected_components(skeleton, connectivity=8)
    component_ids, counts = np.unique(skeleton_components, return_counts=True)
    nonzero = component_ids != 0
    component_ids, counts = component_ids[nonzero], counts[nonzero]
    if len(component_ids) == 0:
        return slice_mask.copy(), slice_mask.copy()
    largest = int(component_ids[np.argmax(counts)])
    skeleton = (skeleton_components == largest).astype(np.uint8)
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 2:
        return slice_mask.copy(), skeleton

    graph = module.nx.Graph()
    for y, x in coordinates:
        node = (int(y), int(x))
        graph.add_node(node)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = (int(y + dy), int(x + dx))
                if (0 <= neighbor[0] < skeleton.shape[0]
                        and 0 <= neighbor[1] < skeleton.shape[1]
                        and skeleton[neighbor]):
                    graph.add_edge(node, neighbor)

    try:
        hull_indices = ConvexHull(coordinates).vertices
    except QhullError:
        hull_indices = np.arange(len(coordinates))
    hull_coordinates = coordinates[hull_indices].astype(np.int64, copy=False)
    differences = hull_coordinates[:, None, :] - hull_coordinates[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    maximum = int(squared_distances.max())
    hull_pairs = np.argwhere(squared_distances == maximum)
    original_pairs = [
        (int(hull_indices[first]), int(hull_indices[second]))
        for first, second in hull_pairs
    ]
    first_index, second_index = min(original_pairs, key=lambda pair: pair[0] * len(coordinates) + pair[1])
    point_a = tuple(int(item) for item in coordinates[first_index])
    point_b = tuple(int(item) for item in coordinates[second_index])
    try:
        path = module.nx.shortest_path(graph, source=point_a, target=point_b)
    except Exception:
        return slice_mask.copy(), skeleton
    path_coordinates = np.asarray(path)
    if len(path_coordinates) > 10:
        start = int(len(path_coordinates) * trunc_fraction)
        end = int(len(path_coordinates) * (1 - trunc_fraction))
        path_coordinates = path_coordinates[start:end]
    scribble = np.zeros_like(slice_mask)
    for y, x in path_coordinates:
        if slice_mask[y, x]:
            scribble[y, x] = 1
    return scribble, skeleton


def _coordinates_to_training_axes(coordinates: list[list[int]], training_to_official_axes):
    inverse_axes = np.argsort(training_to_official_axes)
    return [
        [int(coordinate[inverse_axes[axis]]) for axis in range(len(inverse_axes))]
        for coordinate in coordinates
    ]


def build_cumulative_error_clicks(ground_truth: np.ndarray, champion_prediction: np.ndarray,
                                  num_corrections: int, strategy: str, seed: int = 42,
                                  training_to_official_axes=(2, 1, 0)):
    """Generate 1--5 cumulative prompts from real champion FP/FN regions.

    Each turn follows the official comparison between the candidate foreground
    and background scribble sizes. Since training has no user-in-the-loop model
    execution between turns, the prompted 3-D error component is treated as
    corrected before choosing the next prompt. This prevents duplicate prompts
    and creates a faithful ordered curriculum from the champion's actual errors.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy}")
    if not 1 <= num_corrections <= 5:
        raise ValueError(f"num_corrections must be in [1, 5], got {num_corrections}")
    ground_truth_training = np.asarray(ground_truth, dtype=bool)
    champion_training = np.asarray(champion_prediction, dtype=bool)
    if ground_truth_training.shape != champion_training.shape:
        raise ValueError(f"shape mismatch GT={ground_truth_training.shape} champion={champion_training.shape}")
    training_to_official_axes = tuple(int(axis) for axis in training_to_official_axes)
    if sorted(training_to_official_axes) != list(range(ground_truth_training.ndim)):
        raise ValueError(f"invalid training_to_official_axes={training_to_official_axes}")
    # Official AutoPET V evaluates nibabel arrays in x,y,z order. nnU-Net's
    # Dataset999 tensors are z,y,x because plans.transpose_forward is identity.
    ground_truth = ground_truth_training.transpose(training_to_official_axes)
    working_prediction = champion_training.transpose(training_to_official_axes).copy()

    false_positive = working_prediction & ~ground_truth
    false_negative = ~working_prediction & ground_truth
    fp_labels, fp_candidates = _prepare_error_components(false_positive)
    fn_labels, fn_candidates = _prepare_error_components(false_negative)
    active_fp = set(fp_candidates)
    active_fn = set(fn_candidates)

    clicks = {"tumor": [], "background": []}
    trace = []
    for turn in range(1, num_corrections + 1):
        background, fp_size, fp_component = _best_component_scribble(
            fp_candidates, active_fp, strategy, seed + 2 * turn,
        )
        foreground, fn_size, fn_component = _best_component_scribble(
            fn_candidates, active_fn, strategy, seed + 2 * turn + 1,
        )
        if not background and not foreground:
            trace.append({"turn": turn, "selected": "none", "fp": 0, "fn": 0})
            break
        if foreground and (not background or fp_size <= fn_size):
            clicks["tumor"].extend(_coordinates_to_training_axes(foreground, training_to_official_axes))
            working_prediction[fn_labels == fn_component] = True
            active_fn.remove(fn_component)
            selected = "tumor"
        else:
            clicks["background"].extend(_coordinates_to_training_axes(background, training_to_official_axes))
            working_prediction[fp_labels == fp_component] = False
            active_fp.remove(fp_component)
            selected = "background"
        trace.append({
            "turn": turn,
            "selected": selected,
            "fp_scribble_size": int(fp_size),
            "fn_scribble_size": int(fn_size),
            "remaining_fp_voxels": int(np.count_nonzero(working_prediction & ~ground_truth)),
            "remaining_fn_voxels": int(np.count_nonzero(~working_prediction & ground_truth)),
        })
    return clicks, trace


def build_touched_residual_targets(ground_truth: np.ndarray,
                                   champion_prediction: np.ndarray,
                                   clicks: dict[str, list[list[int]]]) -> np.ndarray:
    """Return ADD/REMOVE targets only for error components touched by a scribble.

    Channel 0 is ``GT & ~M0`` connected to at least one positive point. Channel
    1 is ``M0 & ~GT`` connected to at least one negative point. Consequently,
    errors without a click are explicit KEEP targets instead of asking the
    network to redraw the complete segmentation.
    """
    ground_truth = np.asarray(ground_truth, dtype=bool)
    champion = np.asarray(champion_prediction, dtype=bool)
    if ground_truth.shape != champion.shape:
        raise ValueError(f"shape mismatch GT={ground_truth.shape} champion={champion.shape}")

    residual = np.zeros((2, *ground_truth.shape), dtype=np.uint8)
    specifications = (
        (ground_truth & ~champion, clicks.get("tumor", []), 0),
        (champion & ~ground_truth, clicks.get("background", []), 1),
    )
    for error_mask, points, channel in specifications:
        if not points or not np.any(error_mask):
            continue
        labels = cc3d.connected_components(
            error_mask.astype(np.uint8, copy=False), connectivity=26
        )
        touched = set()
        for point in points:
            coordinate = tuple(int(item) for item in point)
            if len(coordinate) != ground_truth.ndim:
                raise ValueError(f"invalid click coordinate dimensionality: {point}")
            if any(index < 0 or index >= size for index, size in zip(coordinate, ground_truth.shape)):
                raise ValueError(f"click coordinate outside volume: {point} vs {ground_truth.shape}")
            component = int(labels[coordinate])
            if component == 0:
                raise RuntimeError(
                    f"{'positive' if channel == 0 else 'negative'} click {point} "
                    "does not touch its corresponding champion error"
                )
            touched.add(component)
        if touched:
            residual[channel] = np.isin(labels, tuple(touched)).astype(np.uint8)
    return residual
