from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


def pad_nd_image(
    image,
    new_shape,
    mode: str = "constant",
    kwargs: dict | None = None,
    return_slicer: bool = False,
    shape_must_be_divisible_by=None,
):
    """Pad a channel-first image to at least ``new_shape`` along spatial axes."""

    if kwargs is None:
        kwargs = {}

    input_is_torch = isinstance(image, torch.Tensor)
    if input_is_torch:
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    new_shape = np.array(tuple(int(i) for i in new_shape), dtype=int)

    if arr.ndim == len(new_shape):
        spatial_start = 0
    else:
        spatial_start = arr.ndim - len(new_shape)
    spatial_shape = np.array(arr.shape[spatial_start:], dtype=int)

    target_shape = np.maximum(spatial_shape, new_shape)
    if shape_must_be_divisible_by is not None:
        divisor = np.array(shape_must_be_divisible_by, dtype=int)
        if divisor.size == 1:
            divisor = np.repeat(divisor, len(target_shape))
        target_shape = ((target_shape + divisor - 1) // divisor) * divisor

    pad_needed = np.maximum(target_shape - spatial_shape, 0)
    pad_before = pad_needed // 2
    pad_after = pad_needed - pad_before

    pad_spec = [(0, 0)] * spatial_start + [
        (int(before), int(after)) for before, after in zip(pad_before, pad_after)
    ]
    pad_kwargs = dict(kwargs)
    if mode == "constant" and "value" in pad_kwargs and "constant_values" not in pad_kwargs:
        pad_kwargs["constant_values"] = pad_kwargs.pop("value")
    padded = np.pad(arr, pad_spec, mode=mode, **pad_kwargs)
    if input_is_torch:
        padded = torch.as_tensor(padded, dtype=image.dtype)

    if not return_slicer:
        return padded

    slicer = [slice(None)] * spatial_start
    for before, size in zip(pad_before, spatial_shape):
        slicer.append(slice(int(before), int(before + size)))
    return padded, tuple(slicer)
