"""Conservative component-level rejection for PSMA predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import cc3d
import numpy as np


FEATURE_CENTER = np.asarray(
    [
        3.457418291061814, 0.9431892521180966, 0.9727351811893586,
        0.9095072121591514, 0.9681713302998434, 0.9942412730998154,
        0.9317078289910229, 0.992833476324029, 2.4001929120949876,
        1.9738526679112611, 0.4955684463734563, 0.4494882969343918,
        0.5876846004747963,
    ]
)
FEATURE_SCALE = np.asarray(
    [
        1.2412480077852108, 0.7919482402282885, 0.14800346607071332,
        0.19853440725923388, 0.15900513427818647, 0.036981776689039375,
        0.05684951716595382, 0.039490542207436724, 0.7400925283479878,
        0.5835593607945061, 0.11176662594829026, 0.06790542926109018,
        0.22546559424816984,
    ]
)
LINEAR_WEIGHT = np.asarray(
    [
        1.400517179725214, -1.1584763038297488, -0.020826910544214993,
        0.16564773416259684, -0.10267737624230354, -0.3734956749329381,
        0.11979646946766269, 0.34124207067581874, 0.6243423287359192,
        -0.3476595750131606, 0.07496853322613058, 0.15667943427482695,
        -0.03277874136708448,
    ]
)
LINEAR_BIAS = 0.10566988227939351

# Backward-compatible names retained for consumers that inspect the coefficients.
MEAN = FEATURE_CENTER
SCALE = FEATURE_SCALE
COEFFICIENT = LINEAR_WEIGHT
INTERCEPT = LINEAR_BIAS


def _spatial_centroid(component: np.ndarray) -> np.ndarray:
    location = np.asarray(np.nonzero(component), dtype=np.float64).mean(axis=1)
    extent = np.asarray(component.shape, dtype=np.float64)
    return location / np.maximum(extent - 1.0, 1.0)


def _positive_log_stat(values: np.ndarray, reducer) -> float:
    return np.log1p(max(float(reducer(values)), 0.0))


def _features(
    component: np.ndarray,
    probability: np.ndarray,
    pet: np.ndarray,
    spacing: Sequence[float],
) -> np.ndarray:
    voxel_count = int(component.sum())
    volume_ml = voxel_count * float(np.prod(spacing)) / 1000.0
    confidence = probability[component]
    uptake = pet[component]
    confidence_summary = (
        confidence.max(), confidence.mean(), np.quantile(confidence, 0.9)
    )
    return np.asarray(
        [
            np.log1p(voxel_count),
            np.log1p(volume_ml),
            *confidence_summary,
            *confidence_summary,
            _positive_log_stat(uptake, np.max),
            _positive_log_stat(uptake, np.mean),
            *_spatial_centroid(component),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ComponentRejectionModel:
    center: ClassVar[np.ndarray] = FEATURE_CENTER
    scale: ClassVar[np.ndarray] = FEATURE_SCALE
    weight: ClassVar[np.ndarray] = LINEAR_WEIGHT
    bias: ClassVar[float] = LINEAR_BIAS

    def false_probability(self, features: np.ndarray) -> float:
        true_logit = float(((features - self.center) / self.scale) @ self.weight + self.bias)
        return 1.0 - 1.0 / (1.0 + math.exp(-true_logit))


def _validated_inputs(mask, probability, pet):
    binary = np.asarray(mask, dtype=bool)
    confidence = np.asarray(probability, dtype=np.float32)
    uptake = np.asarray(pet, dtype=np.float32)
    if binary.shape != confidence.shape or binary.shape != uptake.shape:
        raise ValueError(
            f"shape mismatch: {binary.shape}, {confidence.shape}, {uptake.shape}"
        )
    return binary, confidence, uptake


def prune_psma_components(
    mask,
    probability,
    pet,
    *,
    spacing,
    false_threshold=0.9,
    false_probability_override: Mapping[int, float] | None = None,
):
    """Remove components whose calibrated false-positive probability is high."""
    binary, confidence, uptake = _validated_inputs(mask, probability, pet)
    component_map, component_count = cc3d.connected_components(
        binary.astype(np.uint8), connectivity=18, return_N=True
    )
    result = binary.copy()
    rejected: list[int] = []
    classifier = ComponentRejectionModel()

    for component_id in range(1, int(component_count) + 1):
        region = component_map == component_id
        if false_probability_override is not None and component_id in false_probability_override:
            false_probability = float(false_probability_override[component_id])
        else:
            descriptor = _features(region, confidence, uptake, spacing)
            false_probability = classifier.false_probability(descriptor)
        if false_probability >= false_threshold:
            result[region] = False
            rejected.append(component_id)

    return result, {
        "input_components": int(component_count),
        "removed_components": len(rejected),
        "false_threshold": float(false_threshold),
    }
