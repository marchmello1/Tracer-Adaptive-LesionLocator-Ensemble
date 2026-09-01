from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def bounding_box_to_slice(bbox: Sequence[Sequence[int]]):
    return tuple(slice(int(start), int(stop)) for start, stop in bbox)


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return [[0, int(s)] for s in mask.shape]
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return [[int(lo[d]), int(hi[d])] for d in range(mask.ndim)]


def insert_crop_into_image(image, crop, bbox: Sequence[Sequence[int]]):
    slicer = bounding_box_to_slice(bbox)
    image[slicer] = crop
    return image


def crop_and_pad_nd(
    image,
    bbox: Sequence[Sequence[int]],
    pad_value=0,
):
    """Crop or pad an array to the requested bounding box.

    nnU-Net mainly uses this with channel-first arrays. The function also works
    for plain spatial arrays.
    """

    arr = np.asarray(image)
    bbox_arr = np.asarray(bbox, dtype=int)
    spatial_dims = bbox_arr.shape[0]
    has_channels = arr.ndim == spatial_dims + 1
    channel_prefix = (slice(None),) if has_channels else ()
    spatial_start = 1 if has_channels else 0

    out_shape = list(arr.shape[:spatial_start]) + [int(hi - lo) for lo, hi in bbox_arr]
    result = np.full(out_shape, pad_value, dtype=arr.dtype)

    src_slices = list(channel_prefix)
    dst_slices = list(channel_prefix)
    for dim, (lo, hi) in enumerate(bbox_arr):
        src_lo = max(int(lo), 0)
        src_hi = min(int(hi), arr.shape[spatial_start + dim])
        width = max(src_hi - src_lo, 0)
        dst_lo = max(-int(lo), 0)
        dst_hi = dst_lo + width
        src_slices.append(slice(src_lo, src_hi))
        dst_slices.append(slice(dst_lo, dst_hi))

    result[tuple(dst_slices)] = arr[tuple(src_slices)]
    return result

