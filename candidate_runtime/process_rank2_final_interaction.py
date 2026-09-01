"""Rank-2 public-champion K0 with validated stateless EDT interaction."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from candidate_runtime.edt_stateless_fusion import fuse_clicked_components
from candidate_runtime.challenge_io import (
    has_effective_scribbles,
    load_request,
    publish_segmentation,
)
from candidate_runtime.initial_prediction import build_initial_prediction


EDT_CODE = Path(os.environ.get("AUTOPET_EDT_CODE", "/opt/algorithm/edt_code"))
EDT_MODEL = Path(os.environ.get("AUTOPET_EDT_MODEL", "/opt/algorithm/edt_model"))
EDT_RUNNER = Path(os.environ.get("AUTOPET_EDT_RUNNER", "/opt/algorithm/edt_runner.py"))

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
    request = load_request(input_root)
    inputs = request.volumes
    scribbles = request.prompts
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
        edt_images = temporary / "edt_images"
        edt_images.mkdir(parents=True)
        initial = build_initial_prediction(ct_image, pet_image, tracer, temporary)
        mask_zyx = initial.mask_zyx
        print(f"initial dust audit={initial.dust_audit}", flush=True)
        if initial.prune_audit is not None:
            print(f"initial PSMA prune audit={initial.prune_audit}", flush=True)

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

    destination = output_root / f"{inputs.identifier}.mha"
    publish_segmentation(mask_zyx, ct_image, destination)
    print(f"output written: {destination}", flush=True)


if __name__ == "__main__":
    process()
