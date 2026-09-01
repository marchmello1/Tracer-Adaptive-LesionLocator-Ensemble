"""Preliminary-only exact public five-fold AutoPET-III champion K0 runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cc3d
import numpy as np


INPUT_ROOT = Path(os.environ.get("AUTOPET_INPUT", "/input"))
OUTPUT_ROOT = Path(
    os.environ.get(
        "AUTOPET_OUTPUT", "/output/images/tumor-lesion-segmentation"
    )
)
CHAMPION_ROOT = Path(
    os.environ.get("AUTOPET_PUBLIC_CHAMPION", "/opt/algorithm/public_champion")
)
CHAMPION_MODEL = Path(
    os.environ.get("AUTOPET_CHAMPION_MODEL", str(CHAMPION_ROOT / "model"))
)
DUST_THRESHOLDS = {"fdg": 25, "psma": 5}


def build_champion_command(input_dir: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(CHAMPION_ROOT / "predict_wrap.py"),
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-d",
        "222",
        "-tr",
        "autoPET3_Trainer",
        "-p",
        "nnUNetResEncUNetLPlansMultiTalent",
        "-c",
        "3d_fullres_bs3",
        "-f",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--disable_tta",
    ]


def apply_tracer_dust(mask: np.ndarray, tracer: str) -> np.ndarray:
    normalized = tracer.lower()
    if normalized not in DUST_THRESHOLDS:
        raise ValueError(f"unsupported tracer: {tracer}")
    binary = np.asarray(mask) > 0
    if not binary.any():
        return binary.astype(np.uint8)
    return (
        cc3d.dust(
            binary.astype(np.uint8),
            threshold=DUST_THRESHOLDS[normalized],
            connectivity=26,
        )
        > 0
    ).astype(np.uint8)


def apply_adaptive_tracer_dust(
    mask: np.ndarray,
    tracer: str,
    *,
    fdg_burden_threshold: int | None,
) -> tuple[np.ndarray, dict[str, int | bool]]:
    robust = apply_tracer_dust(mask, tracer)
    components = int(
        cc3d.connected_components(
            robust.astype(np.uint8), connectivity=26
        ).max()
    )
    relaxed = False
    output = robust
    if tracer.lower() == "fdg" and fdg_burden_threshold is not None:
        if fdg_burden_threshold < 1:
            raise ValueError("FDG burden threshold must be positive")
        if components >= fdg_burden_threshold:
            output = apply_tracer_dust(mask, "psma")
            relaxed = True
    return output, {
        "robust_components": components,
        "fdg_burden_threshold": fdg_burden_threshold or 0,
        "relaxed": relaxed,
    }


def select_tracer_route(
    robust_tracer: str,
    legacy_tracer: str,
    mode: str,
) -> str:
    normalized = mode.strip().lower()
    if normalized == "robust":
        return robust_tracer
    if normalized == "legacy":
        return legacy_tracer
    raise ValueError(f"unsupported public champion tracer mode: {mode}")


def _single_mha(relative_directory: str) -> tuple[Path, str]:
    directory = INPUT_ROOT / relative_directory
    files = sorted(directory.glob("*.mha"))
    if len(files) != 1:
        raise RuntimeError(f"expected one MHA in {directory}, found {files}")
    return files[0], files[0].stem


def process() -> None:
    import SimpleITK as sitk

    from public_tracer_router import predict_tracer

    ct_path, case_id = _single_mha("images/ct")
    pet_path, _ = _single_mha("images/pet")
    ct_image = sitk.ReadImage(str(ct_path))
    pet_image = sitk.ReadImage(str(pet_path))
    robust_tracer, details = predict_tracer(
        sitk.GetArrayFromImage(ct_image),
        sitk.GetArrayFromImage(pet_image),
        return_details=True,
    )
    mode = os.environ.get("AUTOPET_PUBLIC_TRACER_MODE", "robust")
    legacy_tracer = robust_tracer
    legacy_probability = None
    if mode.strip().lower() == "legacy":
        from public_tracer_gate import predict_tracer as predict_legacy_tracer

        legacy_tracer, legacy_probability = predict_legacy_tracer(
            sitk.GetArrayFromImage(ct_image),
            sitk.GetArrayFromImage(pet_image),
            return_prob=True,
        )
    tracer = select_tracer_route(robust_tracer, legacy_tracer, mode)
    print(
        f"public champion tracer={tracer} mode={mode} "
        f"robust_tracer={robust_tracer} legacy_tracer={legacy_tracer} "
        f"legacy_probability={legacy_probability} router={details}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="public_champion_k0_") as temporary:
        temporary = Path(temporary)
        input_dir = temporary / "input"
        output_dir = temporary / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        sitk.WriteImage(ct_image, str(input_dir / "case_0000.nii.gz"), True)
        sitk.WriteImage(pet_image, str(input_dir / "case_0001.nii.gz"), True)
        command = build_champion_command(input_dir, output_dir)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(CHAMPION_ROOT / "autopet-3-submission")
        environment["nnUNet_results"] = str(CHAMPION_MODEL)
        environment["nnUNet_compile"] = "0"
        print("running public champion:", " ".join(command), flush=True)
        subprocess.run(command, env=environment, check=True)
        predicted = sitk.ReadImage(str(output_dir / "case.nii.gz"))
        configured_burden = os.environ.get(
            "AUTOPET_FDG_RELAX_BURDEN_COMPONENTS", ""
        ).strip()
        burden_threshold = int(configured_burden) if configured_burden else None
        mask, dust_audit = apply_adaptive_tracer_dust(
            sitk.GetArrayFromImage(predicted),
            tracer,
            fdg_burden_threshold=burden_threshold,
        )
        print(f"public champion dust audit={dust_audit}", flush=True)
    output_image = sitk.GetImageFromArray(mask)
    if output_image.GetSize() != ct_image.GetSize():
        raise RuntimeError(
            f"output shape mismatch: {output_image.GetSize()} != {ct_image.GetSize()}"
        )
    output_image.CopyInformation(ct_image)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / f"{case_id}.mha"
    sitk.WriteImage(output_image, str(destination), True)
    print(f"output written: {destination}", flush=True)


if __name__ == "__main__":
    process()
