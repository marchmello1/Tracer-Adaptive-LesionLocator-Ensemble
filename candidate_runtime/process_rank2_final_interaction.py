"""Rank-2 public-champion K0 with validated stateless EDT interaction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import SimpleITK as sitk

from candidate_runtime.challenge_io import (
    load_request,
    publish_segmentation,
)
from candidate_runtime.initial_prediction import build_initial_prediction
from candidate_runtime.interactive_update import refine_prediction


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
        initial = build_initial_prediction(ct_image, pet_image, tracer, temporary)
        mask_zyx = initial.mask_zyx
        print(f"initial dust audit={initial.dust_audit}", flush=True)
        if initial.prune_audit is not None:
            print(f"initial PSMA prune audit={initial.prune_audit}", flush=True)

        mask_zyx = refine_prediction(
            mask_zyx, ct_image, pet_image, scribbles, tracer, temporary
        )

    destination = output_root / f"{inputs.identifier}.mha"
    publish_segmentation(mask_zyx, ct_image, destination)
    print(f"output written: {destination}", flush=True)


if __name__ == "__main__":
    process()
