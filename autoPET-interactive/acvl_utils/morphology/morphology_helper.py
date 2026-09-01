from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import ndimage


def label_with_component_sizes(mask: np.ndarray):
    """Return a labeled array and a {component_id: voxel_count} mapping."""

    labeled, num = ndimage.label(mask)
    component_sizes = {}
    if num > 0:
        counts = np.bincount(labeled.ravel())
        for component_id in range(1, len(counts)):
            size = int(counts[component_id])
            if size > 0:
                component_sizes[component_id] = size
    return labeled, component_sizes


def remove_all_but_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, component_sizes = label_with_component_sizes(mask)
    if not component_sizes:
        return np.asarray(mask).astype(bool, copy=False)
    largest_component = max(component_sizes.items(), key=lambda kv: kv[1])[0]
    return labeled == largest_component


def generic_filter_components(mask: np.ndarray, filter_fn: Callable[[Sequence[int], Sequence[int]], Iterable[int]]):
    labeled, component_sizes = label_with_component_sizes(mask)
    if not component_sizes:
        return np.asarray(mask).astype(bool, copy=False)
    component_ids = list(component_sizes.keys())
    sizes = list(component_sizes.values())
    keep_ids = set(filter_fn(component_ids, sizes))
    if not keep_ids:
        return np.zeros_like(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for component_id in keep_ids:
        out |= labeled == component_id
    return out

