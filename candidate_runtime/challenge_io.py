"""Grand Challenge boundary types and deterministic input/output handling."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class VolumeInputs:
    identifier: str
    ct: Path
    pet: Path


@dataclass(frozen=True)
class InteractionRequest:
    volumes: VolumeInputs
    prompts: dict[str, list[list[int]]]


def _unique_mha(folder: Path, role: str) -> Path:
    matches = tuple(sorted(folder.glob("*.mha")))
    if len(matches) != 1:
        raise ValueError(f"expected one {role} MHA in {folder}; found {len(matches)}")
    return matches[0]


def normalize_prompts(document: object) -> dict[str, list[list[int]]]:
    """Convert the challenge point document to bounded integer triplets later."""
    normalized: dict[str, list[list[int]]] = {"tumor": [], "background": []}
    if not isinstance(document, Mapping):
        return normalized
    entries = document.get("points", [])
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return normalized
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        label = entry.get("name")
        coordinates = entry.get("point")
        if label not in normalized or not isinstance(coordinates, list):
            continue
        if len(coordinates) != 3:
            continue
        try:
            normalized[label].append([int(axis) for axis in coordinates])
        except (TypeError, ValueError):
            continue
    return normalized


def load_request(input_root: Path) -> InteractionRequest:
    root = Path(input_root)
    ct = _unique_mha(root / "images" / "ct", "CT")
    pet = _unique_mha(root / "images" / "pet", "PET")
    prompt_file = root / "lesion-clicks.json"
    if not prompt_file.is_file():
        raise ValueError(f"missing lesion-clicks JSON: {prompt_file}")
    document = json.loads(prompt_file.read_text(encoding="utf-8"))
    return InteractionRequest(
        volumes=VolumeInputs(identifier=ct.stem, ct=ct, pet=pet),
        prompts=normalize_prompts(document),
    )


def has_effective_scribbles(prompts: Mapping[str, Sequence]) -> bool:
    """Return whether at least one prompt is an integer-compatible 3-D point."""
    for label in ("tumor", "background"):
        for candidate in prompts.get(label, []):
            if not isinstance(candidate, (list, tuple)) or len(candidate) != 3:
                continue
            try:
                tuple(int(axis) for axis in candidate)
            except (TypeError, ValueError):
                continue
            return True
    return False


def publish_segmentation(mask_zyx: np.ndarray, reference: sitk.Image, destination: Path) -> None:
    result = sitk.GetImageFromArray(np.asarray(mask_zyx, dtype=np.uint8))
    result.CopyInformation(reference)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(result, str(destination), True)

