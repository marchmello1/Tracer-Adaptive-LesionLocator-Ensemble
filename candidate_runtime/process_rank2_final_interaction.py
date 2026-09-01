"""Rank-2 public-champion K0 with validated stateless EDT interaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from candidate_runtime.edt_stateless_fusion import fuse_clicked_components
from candidate_runtime.psma_champion_pruner import prune_psma_components
from candidate_runtime.process_public_champion_k0 import (
    CHAMPION_MODEL,
    CHAMPION_ROOT,
    apply_adaptive_tracer_dust,
    build_champion_command,
)


EDT_CODE = Path(os.environ.get("AUTOPET_EDT_CODE", "/opt/algorithm/edt_code"))
EDT_MODEL = Path(os.environ.get("AUTOPET_EDT_MODEL", "/opt/algorithm/edt_model"))
EDT_RUNNER = Path(os.environ.get("AUTOPET_EDT_RUNNER", "/opt/algorithm/edt_runner.py"))


@dataclass(frozen=True)
class CaseInputs:
    case_id: str
    ct: Path
    pet: Path
    clicks: Path


def _single_mha(directory: Path, label: str) -> Path:
    files = sorted(directory.glob("*.mha"))
    if len(files) != 1:
        raise ValueError(
            f"expected exactly one {label} MHA in {directory}, found {len(files)}"
        )
    return files[0]


def discover_case_inputs(input_root: Path) -> CaseInputs:
    root = Path(input_root)
    ct = _single_mha(root / "images" / "ct", "CT")
    pet = _single_mha(root / "images" / "pet", "PET")
    clicks = root / "lesion-clicks.json"
    if not clicks.is_file():
        raise ValueError(f"missing lesion-clicks JSON: {clicks}")
    return CaseInputs(ct.stem, ct, pet, clicks)


def convert_gc_clicks(click_file: Path) -> dict[str, list[list[int]]]:
    source = json.loads(Path(click_file).read_text(encoding="utf-8"))
    converted: dict[str, list[list[int]]] = {"tumor": [], "background": []}
    for item in source.get("points", []):
        if not isinstance(item, dict):
            continue
        name, point = item.get("name"), item.get("point")
        if name in converted and isinstance(point, list) and len(point) == 3:
            try:
                converted[name].append([int(value) for value in point])
            except (TypeError, ValueError):
                continue
    return converted


def has_effective_scribbles(scribbles: Mapping[str, Sequence]) -> bool:
    for name in ("tumor", "background"):
        for point in scribbles.get(name, []):
            if isinstance(point, (list, tuple)) and len(point) == 3:
                try:
                    [int(value) for value in point]
                except (TypeError, ValueError):
                    continue
                return True
    return False


def _run_edt(images: Path, clicks: Path, output: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(EDT_CODE), "/opt/algorithm"])
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        sys.executable,
        str(EDT_RUNNER),
        "--images", str(images),
        "--clicks", str(clicks),
        "--model", str(EDT_MODEL),
        "--output", str(output),
    ]
    print("running EDT:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    if not output.is_file():
        raise RuntimeError("EDT subprocess did not create its prediction")


def process() -> None:
    from public_tracer_router import predict_tracer

    input_root = Path(os.environ.get("AUTOPET_INPUT", "/input"))
    output_root = Path(
        os.environ.get("AUTOPET_OUTPUT", "/output/images/tumor-lesion-segmentation")
    )
    inputs = discover_case_inputs(input_root)
    scribbles = convert_gc_clicks(inputs.clicks)
    ct_image = sitk.ReadImage(str(inputs.ct))
    pet_image = sitk.ReadImage(str(inputs.pet))
    tracer, details = predict_tracer(
        sitk.GetArrayFromImage(ct_image),
        sitk.GetArrayFromImage(pet_image),
        return_details=True,
    )
    tracer = tracer.strip().lower()
    if tracer not in {"fdg", "psma"}:
        raise ValueError(f"unsupported tracer: {tracer}")
    print(f"rank2-final tracer={tracer} router={details}", flush=True)

    with tempfile.TemporaryDirectory(prefix="rank2_final_") as temporary_name:
        temporary = Path(temporary_name)
        champion_input = temporary / "champion_input"
        champion_output = temporary / "champion_output"
        edt_images = temporary / "edt_images"
        for directory in (champion_input, champion_output, edt_images):
            directory.mkdir(parents=True)
        sitk.WriteImage(ct_image, str(champion_input / "case_0000.nii.gz"), True)
        sitk.WriteImage(pet_image, str(champion_input / "case_0001.nii.gz"), True)
        command = build_champion_command(champion_input, champion_output)
        command.append("--save_probabilities")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(CHAMPION_ROOT / "autopet-3-submission")
        environment["nnUNet_results"] = str(CHAMPION_MODEL)
        environment["nnUNet_compile"] = "0"
        print("running public champion:", " ".join(command), flush=True)
        subprocess.run(command, env=environment, check=True)
        predicted = sitk.ReadImage(str(champion_output / "case.nii.gz"))
        predicted_zyx = sitk.GetArrayFromImage(predicted)
        probability_path = champion_output / "case.npz"
        if not probability_path.is_file():
            raise RuntimeError("public champion did not export probabilities")
        with np.load(probability_path) as archive:
            probability_xyz = np.asarray(archive["probabilities"][1], dtype=np.float32)
        predicted_xyz_shape = np.transpose(predicted_zyx, (2, 1, 0)).shape
        if probability_xyz.shape != predicted_xyz_shape:
            probability_xyz = np.transpose(probability_xyz, (2, 1, 0))
        if probability_xyz.shape != predicted_xyz_shape:
            raise RuntimeError(
                f"probability grid mismatch: {probability_xyz.shape} != {predicted_xyz_shape}"
            )
        raw_zyx = predicted_zyx
        if tracer == "fdg":
            fdg_threshold = float(
                os.environ.get("AUTOPET_FDG_PROBABILITY_THRESHOLD", "0.5")
            )
            raw_zyx = np.transpose(probability_xyz >= fdg_threshold, (2, 1, 0))
        configured_burden = os.environ.get("AUTOPET_FDG_RELAX_BURDEN_COMPONENTS", "25")
        mask_zyx, audit = apply_adaptive_tracer_dust(
            raw_zyx,
            tracer,
            fdg_burden_threshold=int(configured_burden),
        )
        print(f"rank2 K0 dust audit={audit}", flush=True)

        if tracer == "psma":
            mask_xyz = np.transpose(np.asarray(mask_zyx, dtype=bool), (2, 1, 0))
            if probability_xyz.shape != mask_xyz.shape:
                raise RuntimeError(
                    f"PSMA probability grid mismatch: {probability_xyz.shape} != {mask_xyz.shape}"
                )
            pet_xyz = np.transpose(sitk.GetArrayFromImage(pet_image), (2, 1, 0))
            mask_xyz, prune_audit = prune_psma_components(
                mask_xyz,
                probability_xyz,
                pet_xyz,
                spacing=tuple(float(value) for value in pet_image.GetSpacing()),
                false_threshold=float(
                    os.environ.get("AUTOPET_PSMA_PRUNE_THRESHOLD", "0.9")
                ),
            )
            mask_zyx = np.transpose(mask_xyz, (2, 1, 0)).astype(np.uint8)
            print(f"PSMA champion-pruner audit={prune_audit}", flush=True)

        if has_effective_scribbles(scribbles):
            sitk.WriteImage(ct_image, str(edt_images / "TCIA_001_0000.nii.gz"), True)
            sitk.WriteImage(pet_image, str(edt_images / "TCIA_001_0001.nii.gz"), True)
            click_path = temporary / "clicks.json"
            click_path.write_text(json.dumps(scribbles), encoding="utf-8")
            edt_output = temporary / "edt.nii.gz"
            _run_edt(edt_images, click_path, edt_output)
            initial_xyz = np.transpose(np.asarray(mask_zyx, dtype=bool), (2, 1, 0))
            donor_xyz = np.asarray(nib.load(str(edt_output)).dataobj) > 0
            fused_xyz = fuse_clicked_components(
                initial_xyz,
                donor_xyz,
                scribbles,
                tracer,
                disable_background_edits=True,
                # The topology-safe point eraser is PSMA-only. Broad FDG
                # validation improved early Dice but changed later click
                # polarity and regressed AUC on one case.
                certified_background_points=tracer == "psma",
                certified_tumor_points=True,
                consensus_background_deletion=(
                    os.environ.get("AUTOPET_BACKGROUND_CONSENSUS", "0") == "1"
                ),
                background_max_retained_fraction=float(
                    os.environ.get(
                        "AUTOPET_BACKGROUND_MAX_RETAINED_FRACTION", "0.5"
                    )
                ),
                psma_max_components=int(
                    os.environ.get("AUTOPET_PSMA_INTERACTION_MAX_COMPONENTS", "128")
                ),
            )
            mask_zyx = np.transpose(fused_xyz, (2, 1, 0)).astype(np.uint8)

    output_image = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    output_image.CopyInformation(ct_image)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / f"{inputs.case_id}.mha"
    sitk.WriteImage(output_image, str(destination), True)
    print(f"output written: {destination}", flush=True)


if __name__ == "__main__":
    process()
