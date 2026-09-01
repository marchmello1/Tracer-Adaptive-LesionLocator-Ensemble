"""Topology-preserving fusion of automatic masks and interactive corrections.

Prompt validation, evidence selection, and topology checks are deliberately
separate so deployment and tests exercise one explicit interaction policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cc3d
import numpy as np


GridShape = tuple[int, int, int]


def _as_binary_grid(value: np.ndarray) -> np.ndarray:
    grid = np.asarray(value, dtype=bool)
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, received shape {grid.shape}")
    return grid


def _labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = cc3d.connected_components(
        np.asarray(mask, dtype=np.uint8), connectivity=18, return_N=True
    )
    return labels, int(count)


def _parse_points(values: Sequence, shape: GridShape) -> np.ndarray:
    accepted: list[list[int]] = []
    for raw in values:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        try:
            point = [int(coordinate) for coordinate in raw]
        except (TypeError, ValueError):
            continue
        if all(0 <= point[axis] < shape[axis] for axis in range(3)):
            accepted.append(point)
    return np.asarray(accepted, dtype=np.int64).reshape(-1, 3)


def _point_index(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(points[:, axis] for axis in range(3))


def _selected_labels(label_map: np.ndarray, points: np.ndarray) -> list[int]:
    if len(points) == 0:
        return []
    values = np.unique(label_map[_point_index(points)])
    return sorted(int(value) for value in values if value > 0)


def _stroke_masks(points: np.ndarray, shape: GridShape):
    if len(points) == 0:
        return
    prompt_grid = np.zeros(shape, dtype=np.uint8)
    prompt_grid[_point_index(points)] = 1
    stroke_map, stroke_count = _labels(prompt_grid)
    for stroke_id in range(1, stroke_count + 1):
        yield stroke_map == stroke_id


@dataclass(frozen=True)
class PromptSet:
    """Validated foreground and background coordinates on one image grid."""

    tumor: np.ndarray
    background: np.ndarray

    @classmethod
    def from_mapping(cls, source: Mapping[str, Sequence], shape: GridShape):
        return cls(
            tumor=_parse_points(source.get("tumor", []), shape),
            background=_parse_points(source.get("background", []), shape),
        )


@dataclass(frozen=True)
class FusionPolicy:
    tracer: str
    disable_background_edits: bool
    certified_background_points: bool
    certified_tumor_points: bool
    psma_max_components: int
    consensus_background_deletion: bool
    background_max_retained_fraction: float

    @classmethod
    def create(
        cls,
        tracer: str,
        *,
        disable_background_edits: bool,
        certified_background_points: bool,
        certified_tumor_points: bool,
        psma_max_components: int,
        consensus_background_deletion: bool = False,
        background_max_retained_fraction: float = 0.5,
    ):
        normalized = tracer.strip().lower()
        if normalized not in {"fdg", "psma"}:
            raise ValueError(f"Unsupported tracer: {normalized}")
        retained_fraction = float(background_max_retained_fraction)
        if not 0.0 <= retained_fraction <= 1.0:
            raise ValueError("Background retained fraction must be between zero and one")
        return cls(
            tracer=normalized,
            disable_background_edits=disable_background_edits,
            certified_background_points=certified_background_points,
            certified_tumor_points=certified_tumor_points,
            psma_max_components=int(psma_max_components),
            consensus_background_deletion=consensus_background_deletion,
            background_max_retained_fraction=retained_fraction,
        )

    def propagate_tumor(self, prompts: PromptSet, component_count: int) -> bool:
        if len(prompts.tumor) == 0:
            return False
        if self.tracer == "fdg":
            return True
        return len(prompts.tumor) >= 6 and component_count <= self.psma_max_components


class TopologyLedger:
    """Mutable mask whose additions and removals obey component-count guards."""

    def __init__(self, mask: np.ndarray):
        self.mask = mask.copy()
        _, self.component_count = _labels(self.mask)

    def add_if_nonmerging(self, addition: np.ndarray) -> bool:
        candidate = self.mask | addition
        _, candidate_count = _labels(candidate)
        if candidate_count < self.component_count:
            return False
        self.mask = candidate
        self.component_count = candidate_count
        return True

    def remove_if_nonfragmenting(self, removal: np.ndarray) -> bool:
        candidate = self.mask.copy()
        candidate[removal] = False
        _, candidate_count = _labels(candidate)
        if candidate_count > self.component_count:
            return False
        self.mask = candidate
        self.component_count = candidate_count
        return True


def _propagate_clicked_tumor_components(
    ledger: TopologyLedger, donor: np.ndarray, tumor_points: np.ndarray
) -> None:
    donor_labels, _ = _labels(donor)
    for component_id in _selected_labels(donor_labels, tumor_points):
        ledger.add_if_nonmerging(donor_labels == component_id)


def _replace_clicked_background_components(
    ledger: TopologyLedger, donor: np.ndarray, background_points: np.ndarray
) -> None:
    current_labels, _ = _labels(ledger.mask)
    chosen = _selected_labels(current_labels, background_points)
    if not chosen:
        return
    candidate = ledger.mask.copy()
    for component_id in chosen:
        region = current_labels == component_id
        candidate[region] = donor[region]
    _, candidate_count = _labels(candidate)
    if candidate_count <= ledger.component_count:
        ledger.mask = candidate
        ledger.component_count = candidate_count


def _apply_supervised_strokes(
    ledger: TopologyLedger, points: np.ndarray, *, foreground: bool
) -> None:
    for stroke in _stroke_masks(points, ledger.mask.shape):
        if foreground:
            if np.all(ledger.mask[stroke]):
                continue
            ledger.add_if_nonmerging(stroke)
        elif np.any(ledger.mask & stroke):
            ledger.remove_if_nonfragmenting(stroke)


def _apply_consensus_background_deletions(
    ledger: TopologyLedger,
    donor: np.ndarray,
    prompts: PromptSet,
    *,
    max_retained_fraction: float,
) -> None:
    """Delete only components for which prompts and donor independently agree.

    A negative prompt identifies the candidate object, while the interactive
    donor supplies an independent retention vote. Any component containing a
    foreground prompt is protected. Ambiguous objects fall through to the
    conservative stroke-level correction.
    """
    component_map, _ = _labels(ledger.mask)
    protected = set(_selected_labels(component_map, prompts.tumor))
    candidates = _selected_labels(component_map, prompts.background)
    for component_id in candidates:
        if component_id in protected:
            continue
        component = component_map == component_id
        retained = int(np.count_nonzero(donor & component))
        retained_fraction = retained / int(np.count_nonzero(component))
        if retained_fraction < max_retained_fraction:
            ledger.remove_if_nonfragmenting(component)


def fuse_clicked_components(
    initial: np.ndarray,
    edt_prediction: np.ndarray,
    scribbles: Mapping[str, Sequence],
    tracer: str,
    *,
    disable_background_edits: bool = False,
    certified_background_points: bool = False,
    certified_tumor_points: bool = False,
    psma_max_components: int = 128,
    consensus_background_deletion: bool = False,
    background_max_retained_fraction: float = 0.5,
) -> np.ndarray:
    """Return a stateless, topology-safe fusion for cumulative interaction."""
    accepted = _as_binary_grid(initial)
    donor = _as_binary_grid(edt_prediction)
    if accepted.shape != donor.shape:
        raise ValueError(f"Initial/EDT grid mismatch: {accepted.shape} != {donor.shape}")

    policy = FusionPolicy.create(
        tracer,
        disable_background_edits=disable_background_edits,
        certified_background_points=certified_background_points,
        certified_tumor_points=certified_tumor_points,
        psma_max_components=psma_max_components,
        consensus_background_deletion=consensus_background_deletion,
        background_max_retained_fraction=background_max_retained_fraction,
    )
    prompts = PromptSet.from_mapping(scribbles, accepted.shape)
    ledger = TopologyLedger(accepted)

    if policy.propagate_tumor(prompts, ledger.component_count):
        _propagate_clicked_tumor_components(ledger, donor, prompts.tumor)

    if (
        policy.tracer == "fdg"
        and len(prompts.background) > 0
        and not policy.disable_background_edits
    ):
        _replace_clicked_background_components(ledger, donor, prompts.background)

    if policy.consensus_background_deletion and len(prompts.background) > 0:
        _apply_consensus_background_deletions(
            ledger,
            donor,
            prompts,
            max_retained_fraction=policy.background_max_retained_fraction,
        )

    if policy.certified_background_points:
        _apply_supervised_strokes(ledger, prompts.background, foreground=False)
    if policy.certified_tumor_points:
        _apply_supervised_strokes(ledger, prompts.tumor, foreground=True)
    return ledger.mask
