"""Run the LesionLocator EDT fold-0 model in an isolated Python subprocess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.autopet_predictor import autoPETPredictor


def gc_payload(scribbles: dict) -> dict:
    points = []
    for name in ("tumor", "background"):
        for point in scribbles.get(name, []):
            points.append({"point": point, "name": name})
    return {
        "version": {"major": 1, "minor": 0},
        "type": "Multiple points",
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--clicks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ct_path = args.images / "TCIA_001_0000.nii.gz"
    pet_path = args.images / "TCIA_001_0001.nii.gz"
    ct_image = sitk.ReadImage(str(ct_path))
    pet_image = sitk.ReadImage(str(pet_path))
    image = np.stack(
        [sitk.GetArrayFromImage(ct_image), sitk.GetArrayFromImage(pet_image)]
    ).astype(np.float16)
    properties = {"spacing": list(ct_image.GetSpacing())[::-1]}
    clicks = gc_payload(json.loads(args.clicks.read_text()))
    predictor = autoPETPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda:0"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model, use_folds=(0,), checkpoint_name="checkpoint_final.pth"
    )
    prediction_zyx = predictor.predict_single_npy_array(
        image, properties, clicks, 2.0, None, None, False
    )
    prediction_xyz = np.transpose(np.asarray(prediction_zyx), (2, 1, 0)) > 0
    reference = nib.load(str(ct_path))
    output = nib.Nifti1Image(
        prediction_xyz.astype(np.uint8), reference.affine, reference.header.copy()
    )
    output.set_data_dtype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(args.output))


if __name__ == "__main__":
    main()
