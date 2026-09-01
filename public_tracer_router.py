"""Robust image-only FDG/PSMA router for the production container."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tracer_router_ensemble_v1.npz"
)
PET_MIN = -5.0
PET_MAX = 500.0
CT_MIN = -2_000.0
CT_MAX = 10_000.0
N_AXIAL_BINS = 12


def _safe_percentile(values, percentile):
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _sanitize(raw, low, high):
    raw = np.asarray(raw)
    finite = np.isfinite(raw)
    plausible = finite & (raw >= low) & (raw <= high)
    stats = {
        "finite_fraction": float(finite.mean()),
        "plausible_fraction": float(plausible.mean()),
        "invalid_count": int(raw.size - np.count_nonzero(plausible)),
    }
    cleaned = np.where(plausible, raw, 0.0).astype(np.float32, copy=False)
    return cleaned, stats


def _axial_body_bounds(body):
    per_slice = body.reshape(body.shape[0], -1).mean(axis=1)
    occupied = np.flatnonzero(per_slice > 0.01)
    if occupied.size == 0:
        return 0, body.shape[0]
    return int(occupied[0]), int(occupied[-1]) + 1


def extract_router_features(ct_raw, pet_raw):
    """Extract the fixed 99-feature routing representation from z-y-x arrays."""
    ct_raw = np.asarray(ct_raw)[::2, ::2, ::2]
    pet_raw = np.asarray(pet_raw)[::2, ::2, ::2]
    if ct_raw.shape != pet_raw.shape:
        raise RuntimeError(f"tracer-router grid mismatch: {ct_raw.shape} != {pet_raw.shape}")

    ct, ct_stats = _sanitize(ct_raw, CT_MIN, CT_MAX)
    pet, pet_stats = _sanitize(pet_raw, PET_MIN, PET_MAX)
    body = (ct > -500.0) & (ct < 3_000.0)
    z0, z1 = _axial_body_bounds(body)
    positive = pet[body & (pet > 0.1)]
    if positive.size == 0:
        positive = pet[pet > 0.0]

    names = []
    values = []

    def add(name, value):
        names.append(name)
        values.append(float(value))

    for q in (50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9):
        add(f"global_p{str(q).replace('.', '_')}", _safe_percentile(positive, q))
    add("global_mean", float(positive.mean()) if positive.size else 0.0)
    add("global_std", float(positive.std()) if positive.size else 0.0)
    body_count = max(int(body.sum()), 1)
    for threshold in (1.0, 2.5, 4.0, 10.0, 20.0):
        fraction = float(((pet > threshold) & body).sum()) / body_count
        add(f"global_frac_gt_{str(threshold).replace('.', '_')}", fraction)

    add("pet_finite_fraction", pet_stats["finite_fraction"])
    add("pet_plausible_fraction", pet_stats["plausible_fraction"])
    add("ct_finite_fraction", ct_stats["finite_fraction"])
    add("ct_plausible_fraction", ct_stats["plausible_fraction"])
    add("body_z_fraction", float(z1 - z0) / max(pet.shape[0], 1))

    edges = np.linspace(z0, z1, N_AXIAL_BINS + 1, dtype=int)
    bin_mass = []
    bin_p99 = []
    bin_bone = []
    for index in range(N_AXIAL_BINS):
        lo, hi = int(edges[index]), int(edges[index + 1])
        if hi <= lo:
            hi = min(lo + 1, pet.shape[0])
        region_body = body[lo:hi]
        region_pet = pet[lo:hi]
        region_ct = ct[lo:hi]
        region_positive = region_pet[region_body & (region_pet > 0.1)]
        denominator = max(int(region_body.sum()), 1)
        p90 = _safe_percentile(region_positive, 90.0)
        p99 = _safe_percentile(region_positive, 99.0)
        mean = float(region_positive.mean()) if region_positive.size else 0.0
        frac25 = float(((region_pet > 2.5) & region_body).sum()) / denominator
        frac4 = float(((region_pet > 4.0) & region_body).sum()) / denominator
        bone = float(((region_ct > 300.0) & region_body).sum()) / denominator
        mass = float(np.clip(region_pet, 0.0, None)[region_body].sum())
        for suffix, value in (
            ("p90", p90),
            ("p99", p99),
            ("mean", mean),
            ("frac_gt_2_5", frac25),
            ("frac_gt_4", frac4),
            ("bone_frac", bone),
        ):
            add(f"zbin_{index:02d}_{suffix}", value)
        bin_mass.append(mass)
        bin_p99.append(p99)
        bin_bone.append(bone)

    total_mass = max(sum(bin_mass), 1e-8)
    normalized_mass = np.asarray(bin_mass, dtype=np.float64) / total_mass
    coordinates = (np.arange(N_AXIAL_BINS, dtype=np.float64) + 0.5) / N_AXIAL_BINS
    centroid = float(np.dot(normalized_mass, coordinates))
    spread = float(np.sqrt(np.dot(normalized_mass, (coordinates - centroid) ** 2)))
    entropy = float(-(normalized_mass * np.log(normalized_mass + 1e-12)).sum())
    add("axial_uptake_centroid", centroid)
    add("axial_uptake_spread", spread)
    add("axial_uptake_entropy", entropy)

    endpoint_p99 = (bin_p99[0], bin_p99[-1])
    endpoint_bone = (bin_bone[0], bin_bone[-1])
    mid_p99 = float(np.median(bin_p99[4:8])) + 1e-6
    add("endpoint_p99_min", min(endpoint_p99))
    add("endpoint_p99_max", max(endpoint_p99))
    add("endpoint_p99_max_to_mid", max(endpoint_p99) / mid_p99)
    add("endpoint_bone_min", min(endpoint_bone))
    add("endpoint_bone_max", max(endpoint_bone))

    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite tracer-router feature after sanitization")
    return features, names, {
        "pet_invalid_count": pet_stats["invalid_count"],
        "ct_invalid_count": ct_stats["invalid_count"],
    }


@lru_cache(maxsize=1)
def _load_model(model_path):
    archive = np.load(model_path, allow_pickle=False)
    payload = {name: archive[name] for name in archive.files}
    if int(payload["format_version"][0]) != 1:
        raise RuntimeError("unsupported tracer-router model version")
    return payload


def _impute(x, statistics):
    return np.where(np.isnan(x), statistics, x)


def _predict_hgb(model, x):
    row = _impute(x, model["hgb_imputer"])
    score = float(model["hgb_baseline"][0])
    offsets = model["hgb_offsets"]
    for tree_index in range(len(offsets) - 1):
        start = int(offsets[tree_index])
        node = start
        while not model["hgb_leaf"][node]:
            feature = int(model["hgb_feature"][node])
            value = row[feature]
            if np.isnan(value):
                go_left = bool(model["hgb_missing_left"][node])
            else:
                go_left = value <= model["hgb_threshold"][node]
            child = model["hgb_left"][node] if go_left else model["hgb_right"][node]
            node = start + int(child)
        score += float(model["hgb_value"][node])
    return int(score > 0.0)


def _predict_svm(model, x):
    row = _impute(x, model["svm_imputer"])
    row = (row - model["svm_mean"]) / model["svm_scale"]
    score = float(row @ model["svm_coef"] + model["svm_intercept"][0])
    return int(score > 0.0)


def _predict_forest(model, x):
    row = _impute(x, model["forest_imputer"])
    offsets = model["forest_offsets"]
    probability_sum = 0.0
    for tree_index in range(len(offsets) - 1):
        start = int(offsets[tree_index])
        node = start
        while model["forest_left"][node] != -1:
            feature = int(model["forest_feature"][node])
            child = (
                model["forest_left"][node]
                if row[feature] <= model["forest_threshold"][node]
                else model["forest_right"][node]
            )
            node = start + int(child)
        probability_sum += float(model["forest_leaf_p1"][node])
    return int(probability_sum / (len(offsets) - 1) > 0.5)


def predict_tracer(ct, pet, model_path=None, return_details=False):
    """Predict the tracer by a fixed unweighted three-model majority vote."""
    payload = _load_model(model_path or MODEL_PATH)
    features, names, input_stats = extract_router_features(ct, pet)
    if names != payload["feature_names"].tolist():
        raise RuntimeError("tracer-router feature schema does not match model")
    raw_votes = {
        "hgb": _predict_hgb(payload, features),
        "linear_svm": _predict_svm(payload, features),
        "random_forest": _predict_forest(payload, features),
    }
    psma_votes = int(sum(raw_votes.values()))
    label = "psma" if psma_votes >= 2 else "fdg"
    details = {
        "psma_votes": psma_votes,
        "votes": {name: ("psma" if value else "fdg") for name, value in raw_votes.items()},
        "unanimous": len(set(raw_votes.values())) == 1,
        **input_stats,
    }
    return (label, details) if return_details else label
