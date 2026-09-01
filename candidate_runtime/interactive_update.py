"""Interactive donor inference and topology-safe reconciliation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from candidate_runtime.challenge_io import has_effective_scribbles
from candidate_runtime.edt_stateless_fusion import fuse_clicked_components


@dataclass(frozen=True)
class InteractiveBackend:
    source: Path
    weights: Path
    runner: Path

    @classmethod
    def from_environment(cls) -> "InteractiveBackend":
        return cls(
            source=Path(os.environ.get("AUTOPET_EDT_CODE", "/opt/algorithm/edt_code")),
            weights=Path(os.environ.get("AUTOPET_EDT_MODEL", "/opt/algorithm/edt_model")),
            runner=Path(
                os.environ.get("AUTOPET_EDT_RUNNER", "/opt/algorithm/edt_runner.py")
            ),
        )


@dataclass(frozen=True)
class InteractionPolicy:
    psma_component_ceiling: int
    consensus_background: bool
    background_retention_limit: float

    @classmethod
    def from_environment(cls) -> "InteractionPolicy":
        return cls(
            psma_component_ceiling=int(
                os.environ.get("AUTOPET_PSMA_INTERACTION_MAX_COMPONENTS", "128")
            ),
            consensus_background=(
                os.environ.get("AUTOPET_BACKGROUND_CONSENSUS", "0") == "1"
            ),
            background_retention_limit=float(
                os.environ.get("AUTOPET_BACKGROUND_MAX_RETAINED_FRACTION", "0.5")
            ),
        )


def _infer_donor(
    ct_image: sitk.Image,
    pet_image: sitk.Image,
    prompts: Mapping[str, Sequence],
    workspace: Path,
    backend: InteractiveBackend,
) -> np.ndarray:
    image_folder = workspace / "interaction_input"
    image_folder.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(ct_image, str(image_folder / "TCIA_001_0000.nii.gz"), True)
    sitk.WriteImage(pet_image, str(image_folder / "TCIA_001_0001.nii.gz"), True)
    prompt_path = workspace / "interaction_prompts.json"
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    output_path = workspace / "interaction_donor.nii.gz"

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(backend.source), "/opt/algorithm"]
    )
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable,
        str(backend.runner),
        "--images",
        str(image_folder),
        "--clicks",
        str(prompt_path),
        "--model",
        str(backend.weights),
        "--output",
        str(output_path),
    ]
    print("running interactive donor:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    if not output_path.is_file():
        raise RuntimeError("interactive predictor did not create its output")
    return np.asarray(nib.load(str(output_path)).dataobj) > 0


def refine_prediction(
    initial_zyx: np.ndarray,
    ct_image: sitk.Image,
    pet_image: sitk.Image,
    prompts: Mapping[str, Sequence],
    tracer: str,
    workspace: Path,
    *,
    backend: InteractiveBackend | None = None,
    policy: InteractionPolicy | None = None,
) -> np.ndarray:
    """Return the initial mask or a safely reconciled interactive proposal."""
    if not has_effective_scribbles(prompts):
        return np.asarray(initial_zyx, dtype=np.uint8)
    selected_backend = backend or InteractiveBackend.from_environment()
    selected_policy = policy or InteractionPolicy.from_environment()
    donor_xyz = _infer_donor(
        ct_image, pet_image, prompts, workspace, selected_backend
    )
    initial_xyz = np.transpose(np.asarray(initial_zyx, dtype=bool), (2, 1, 0))
    if donor_xyz.shape != initial_xyz.shape:
        raise RuntimeError(
            f"interactive donor grid mismatch: {donor_xyz.shape} != {initial_xyz.shape}"
        )
    normalized_tracer = tracer.strip().lower()
    fused_xyz = fuse_clicked_components(
        initial_xyz,
        donor_xyz,
        prompts,
        normalized_tracer,
        disable_background_edits=True,
        certified_background_points=normalized_tracer == "psma",
        certified_tumor_points=True,
        consensus_background_deletion=selected_policy.consensus_background,
        background_max_retained_fraction=selected_policy.background_retention_limit,
        psma_max_components=selected_policy.psma_component_ceiling,
    )
    return np.transpose(fused_xyz, (2, 1, 0)).astype(np.uint8)
