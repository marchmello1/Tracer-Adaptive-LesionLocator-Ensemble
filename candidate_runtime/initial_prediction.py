"""Tracer-conditioned initial segmentation assembled from frozen predictors."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from candidate_runtime.process_public_champion_k0 import (
    CHAMPION_MODEL,
    CHAMPION_ROOT,
    apply_adaptive_tracer_dust,
    build_champion_command,
)
from candidate_runtime.psma_champion_pruner import prune_psma_components


@dataclass(frozen=True)
class InitialPredictionConfig:
    fdg_probability_threshold: float
    fdg_relax_burden_components: int
    psma_prune_threshold: float

    @classmethod
    def from_environment(cls) -> "InitialPredictionConfig":
        return cls(
            fdg_probability_threshold=float(
                os.environ.get("AUTOPET_FDG_PROBABILITY_THRESHOLD", "0.47")
            ),
            fdg_relax_burden_components=int(
                os.environ.get("AUTOPET_FDG_RELAX_BURDEN_COMPONENTS", "25")
            ),
            psma_prune_threshold=float(
                os.environ.get("AUTOPET_PSMA_PRUNE_THRESHOLD", "0.86")
            ),
        )


@dataclass(frozen=True)
class InitialPrediction:
    mask_zyx: np.ndarray
    dust_audit: dict
    prune_audit: dict | None


def _load_foreground_probability(archive_path: Path, expected_zyx: tuple[int, ...]) -> np.ndarray:
    if not archive_path.is_file():
        raise RuntimeError("initial predictor did not export probabilities")
    with np.load(archive_path) as archive:
        probability_xyz = np.asarray(archive["probabilities"][1], dtype=np.float32)
    expected_xyz = tuple(reversed(expected_zyx))
    if probability_xyz.shape != expected_xyz:
        probability_xyz = np.transpose(probability_xyz, (2, 1, 0))
    if probability_xyz.shape != expected_xyz:
        raise RuntimeError(
            f"probability grid mismatch: {probability_xyz.shape} != {expected_xyz}"
        )
    return probability_xyz


def build_initial_prediction(
    ct_image: sitk.Image,
    pet_image: sitk.Image,
    tracer: str,
    workspace: Path,
    config: InitialPredictionConfig | None = None,
) -> InitialPrediction:
    """Run the frozen ensemble and apply tracer-specific conservative cleanup."""
    policy = config or InitialPredictionConfig.from_environment()
    normalized_tracer = tracer.strip().lower()
    if normalized_tracer not in {"fdg", "psma"}:
        raise ValueError(f"unsupported tracer: {normalized_tracer}")

    model_input = workspace / "initial_input"
    model_output = workspace / "initial_output"
    model_input.mkdir(parents=True, exist_ok=True)
    model_output.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_image, str(model_input / "case_0000.nii.gz"), True)
    sitk.WriteImage(pet_image, str(model_input / "case_0001.nii.gz"), True)

    command = build_champion_command(model_input, model_output)
    command.append("--save_probabilities")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CHAMPION_ROOT / "autopet-3-submission")
    environment["nnUNet_results"] = str(CHAMPION_MODEL)
    environment["nnUNet_compile"] = "0"
    print("running initial ensemble:", " ".join(command), flush=True)
    subprocess.run(command, env=environment, check=True)

    predicted = sitk.ReadImage(str(model_output / "case.nii.gz"))
    predicted_zyx = sitk.GetArrayFromImage(predicted)
    probability_xyz = _load_foreground_probability(
        model_output / "case.npz", predicted_zyx.shape
    )
    raw_zyx = predicted_zyx
    if normalized_tracer == "fdg":
        raw_zyx = np.transpose(
            probability_xyz >= policy.fdg_probability_threshold, (2, 1, 0)
        )

    cleaned_zyx, dust_audit = apply_adaptive_tracer_dust(
        raw_zyx,
        normalized_tracer,
        fdg_burden_threshold=policy.fdg_relax_burden_components,
    )
    prune_audit = None
    if normalized_tracer == "psma":
        mask_xyz = np.transpose(np.asarray(cleaned_zyx, dtype=bool), (2, 1, 0))
        pet_xyz = np.transpose(sitk.GetArrayFromImage(pet_image), (2, 1, 0))
        mask_xyz, prune_audit = prune_psma_components(
            mask_xyz,
            probability_xyz,
            pet_xyz,
            spacing=tuple(float(value) for value in pet_image.GetSpacing()),
            false_threshold=policy.psma_prune_threshold,
        )
        cleaned_zyx = np.transpose(mask_xyz, (2, 1, 0)).astype(np.uint8)

    return InitialPrediction(
        mask_zyx=np.asarray(cleaned_zyx, dtype=np.uint8),
        dust_audit=dust_audit,
        prune_audit=prune_audit,
    )

