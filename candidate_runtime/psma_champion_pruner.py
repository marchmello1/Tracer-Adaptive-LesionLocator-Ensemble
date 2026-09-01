"""Champion-only PSMA component false-positive pruning."""

from __future__ import annotations

import math
from collections.abc import Mapping

import cc3d
import numpy as np


MEAN = np.asarray([3.457418291061814, 0.9431892521180966, 0.9727351811893586, 0.9095072121591514, 0.9681713302998434, 0.9942412730998154, 0.9317078289910229, 0.992833476324029, 2.4001929120949876, 1.9738526679112611, 0.4955684463734563, 0.4494882969343918, 0.5876846004747963])
SCALE = np.asarray([1.2412480077852108, 0.7919482402282885, 0.14800346607071332, 0.19853440725923388, 0.15900513427818647, 0.036981776689039375, 0.05684951716595382, 0.039490542207436724, 0.7400925283479878, 0.5835593607945061, 0.11176662594829026, 0.06790542926109018, 0.22546559424816984])
COEFFICIENT = np.asarray([1.400517179725214, -1.1584763038297488, -0.020826910544214993, 0.16564773416259684, -0.10267737624230354, -0.3734956749329381, 0.11979646946766269, 0.34124207067581874, 0.6243423287359192, -0.3476595750131606, 0.07496853322613058, 0.15667943427482695, -0.03277874136708448])
INTERCEPT = 0.10566988227939351


def _features(component, probability, pet, spacing):
    voxels = int(component.sum())
    volume_ml = voxels * float(np.prod(spacing)) / 1000.0
    p = probability[component]
    suv = pet[component]
    shape = np.asarray(component.shape, dtype=np.float64)
    centroid = np.asarray(np.nonzero(component), dtype=np.float64).mean(axis=1)
    centroid /= np.maximum(shape - 1.0, 1.0)
    return np.asarray([
        np.log1p(voxels), np.log1p(volume_ml),
        p.max(), p.mean(), np.quantile(p, 0.9),
        p.max(), p.mean(), np.quantile(p, 0.9),
        np.log1p(max(float(suv.max()), 0.0)),
        np.log1p(max(float(suv.mean()), 0.0)),
        *centroid,
    ], dtype=np.float64)


def prune_psma_components(
    mask,
    probability,
    pet,
    *,
    spacing,
    false_threshold=0.9,
    false_probability_override: Mapping[int, float] | None = None,
):
    mask = np.asarray(mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    pet = np.asarray(pet, dtype=np.float32)
    if mask.shape != probability.shape or mask.shape != pet.shape:
        raise ValueError(f"shape mismatch: {mask.shape}, {probability.shape}, {pet.shape}")
    labels, count = cc3d.connected_components(mask.astype(np.uint8), connectivity=18, return_N=True)
    output = mask.copy()
    removed = []
    for label in range(1, count + 1):
        component = labels == label
        if false_probability_override is not None and label in false_probability_override:
            false_probability = float(false_probability_override[label])
        else:
            features = _features(component, probability, pet, spacing)
            true_logit = float(((features - MEAN) / SCALE) @ COEFFICIENT + INTERCEPT)
            true_probability = 1.0 / (1.0 + math.exp(-true_logit))
            false_probability = 1.0 - true_probability
        if false_probability >= false_threshold:
            output[component] = False
            removed.append(label)
    return output, {
        "input_components": int(count),
        "removed_components": len(removed),
        "false_threshold": float(false_threshold),
    }
