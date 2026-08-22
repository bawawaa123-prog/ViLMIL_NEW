"""Coordinate-value based variable low-to-high patch mapping.

The mapper operates on feature-H5 coordinates, whose rows must remain paired
with the corresponding feature rows. Coordinates are top-left positions in
level-0 pixels. Patch sizes are specified in their native pyramid level and
converted to continuous level-0 footprints with the supplied downsample.

No fixed child count, row-order pairing, or nearest-neighbour assignment is
used. Positive-area bbox overlap creates a relation. All relations are kept
in deterministic CSR arrays, including high patches with no parent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from collections import defaultdict
import math

import numpy as np


def _pair(value: float | Iterable[float], name: str) -> tuple[float, float]:
    if np.isscalar(value):
        result = (float(value), float(value))
    else:
        values = tuple(float(x) for x in value)
        if len(values) != 2:
            raise ValueError(f"{name} must be scalar or length-2 iterable")
        result = values
    if not all(np.isfinite(result)) or min(result) <= 0:
        raise ValueError(f"{name} must contain finite positive values")
    return result


def _coords(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N, 2], got {array.shape}")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


@dataclass(frozen=True)
class CrossScaleMapping:
    """Immutable mapping result for one slide.

    `parent_ptr[i]:parent_ptr[i+1]` indexes high children of low patch `i`.
    The reverse CSR uses `high_parent_ptr` and `parent_indices`. Relations are
    sorted by source index then destination index, so the representation is
    reproducible regardless of candidate-index implementation details.
    """

    low_bboxes: np.ndarray
    high_bboxes: np.ndarray
    low_valid_mask: np.ndarray
    high_valid_mask: np.ndarray
    low_padding_ratio: np.ndarray
    high_padding_ratio: np.ndarray
    parent_ptr: np.ndarray
    child_indices: np.ndarray
    child_overlap_area: np.ndarray
    child_weight: np.ndarray
    child_low_fraction: np.ndarray
    child_high_fraction: np.ndarray
    high_parent_ptr: np.ndarray
    parent_indices: np.ndarray
    parent_overlap_area: np.ndarray
    parent_weight: np.ndarray
    unmapped_high_indices: np.ndarray
    high_has_parent_mask: np.ndarray
    primary_parent: np.ndarray
    primary_parent_overlap_area: np.ndarray

    @property
    def low_child_counts(self) -> np.ndarray:
        return np.diff(self.parent_ptr)

    @property
    def high_parent_counts(self) -> np.ndarray:
        return np.diff(self.high_parent_ptr)

    @property
    def parent_coverage(self) -> float:
        return float(self.high_has_parent_mask.mean()) if len(self.high_has_parent_mask) else 1.0

    @property
    def low_coverage(self) -> float:
        counts = self.low_child_counts
        return float((counts > 0).mean()) if len(counts) else 1.0

    def children(self, low_index: int) -> np.ndarray:
        start, end = self.parent_ptr[low_index : low_index + 2]
        return self.child_indices[start:end]

    def parents(self, high_index: int) -> np.ndarray:
        start, end = self.high_parent_ptr[high_index : high_index + 2]
        return self.parent_indices[start:end]

    def validate(self) -> None:
        """Check CSR bounds, weights, and complete high-patch coverage."""
        n_low = len(self.low_bboxes)
        n_high = len(self.high_bboxes)
        if self.parent_ptr.shape != (n_low + 1,):
            raise ValueError("parent_ptr must have length n_low + 1")
        if self.high_parent_ptr.shape != (n_high + 1,):
            raise ValueError("high_parent_ptr must have length n_high + 1")
        if self.parent_ptr[0] != 0 or self.high_parent_ptr[0] != 0:
            raise ValueError("CSR pointers must start at zero")
        if self.parent_ptr[-1] != len(self.child_indices):
            raise ValueError("parent_ptr does not cover child_indices")
        if self.high_parent_ptr[-1] != len(self.parent_indices):
            raise ValueError("high_parent_ptr does not cover parent_indices")
        if len(self.child_indices) and (self.child_indices.min() < 0 or self.child_indices.max() >= n_high):
            raise ValueError("child_indices contains an out-of-range high index")
        if len(self.parent_indices) and (self.parent_indices.min() < 0 or self.parent_indices.max() >= n_low):
            raise ValueError("parent_indices contains an out-of-range low index")
        expected_unmapped = np.flatnonzero(~self.high_has_parent_mask)
        if not np.array_equal(expected_unmapped, self.unmapped_high_indices):
            raise ValueError("unmapped_high_indices disagrees with high_has_parent_mask")
        for start, end in zip(self.parent_ptr[:-1], self.parent_ptr[1:]):
            if end > start and not np.isclose(self.child_weight[start:end].sum(), 1.0):
                raise ValueError("child weights are not normalized per low parent")

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a mapping payload suitable for a dataset sample."""
        return {name: getattr(self, name) for name in _SERIALIZED_FIELDS}

    def save_npz(self, path: str) -> None:
        """Persist all CSR, weight, bbox, and mask arrays without reordering."""
        np.savez_compressed(path, **{name: getattr(self, name) for name in _SERIALIZED_FIELDS})

    @classmethod
    def load_npz(cls, path: str) -> "CrossScaleMapping":
        with np.load(path, allow_pickle=False) as archive:
            missing = [name for name in _SERIALIZED_FIELDS if name not in archive]
            if missing:
                raise ValueError(f"mapping cache missing fields: {missing}")
            result = cls(**{name: archive[name] for name in _SERIALIZED_FIELDS})
        result.validate()
        return result


_SERIALIZED_FIELDS = (
    "low_bboxes", "high_bboxes", "low_valid_mask", "high_valid_mask",
    "low_padding_ratio", "high_padding_ratio", "parent_ptr", "child_indices",
    "child_overlap_area", "child_weight", "child_low_fraction",
    "child_high_fraction", "high_parent_ptr", "parent_indices",
    "parent_overlap_area", "parent_weight", "unmapped_high_indices",
    "high_has_parent_mask", "primary_parent", "primary_parent_overlap_area",
)


def _make_bboxes(
    coords: np.ndarray,
    patch_size: float | Iterable[float],
    downsample: float | Iterable[float],
) -> np.ndarray:
    size_x, size_y = _pair(patch_size, "patch_size")
    ds_x, ds_y = _pair(downsample, "downsample")
    width, height = size_x * ds_x, size_y * ds_y
    boxes = np.empty((len(coords), 4), dtype=np.float64)
    boxes[:, :2] = coords
    boxes[:, 2] = coords[:, 0] + width
    boxes[:, 3] = coords[:, 1] + height
    return boxes


def _padding_ratio(boxes: np.ndarray, wsi_dimensions: tuple[float, float] | None) -> np.ndarray:
    if wsi_dimensions is None:
        return np.zeros(len(boxes), dtype=np.float64)
    wsi_w, wsi_h = _pair(wsi_dimensions, "wsi_dimensions")
    inside_w = np.maximum(0.0, np.minimum(boxes[:, 2], wsi_w) - np.maximum(boxes[:, 0], 0.0))
    inside_h = np.maximum(0.0, np.minimum(boxes[:, 3], wsi_h) - np.maximum(boxes[:, 1], 0.0))
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return np.clip(1.0 - (inside_w * inside_h) / area, 0.0, 1.0)


def _valid_mask(boxes: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(boxes).all(axis=1)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )


def _candidate_index(boxes: np.ndarray, bin_width: float, bin_height: float) -> dict[tuple[int, int], list[int]]:
    """Index high boxes by low-sized spatial bins in deterministic order."""
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for high_index, box in enumerate(boxes):
        x0 = math.floor(box[0] / bin_width)
        x1 = math.floor(np.nextafter(box[2], -np.inf) / bin_width)
        y0 = math.floor(box[1] / bin_height)
        y1 = math.floor(np.nextafter(box[3], -np.inf) / bin_height)
        for bx in range(x0, x1 + 1):
            for by in range(y0, y1 + 1):
                index[(bx, by)].append(high_index)
    return index


def build_cross_scale_mapping(
    low_coords: np.ndarray,
    high_coords: np.ndarray,
    *,
    low_patch_size: float | Iterable[float],
    high_patch_size: float | Iterable[float],
    low_downsample: float | Iterable[float],
    high_downsample: float | Iterable[float],
    wsi_dimensions: tuple[float, float] | None = None,
) -> CrossScaleMapping:
    """Build deterministic positive-area bbox overlap mapping.

    A parent is any low bbox with positive-area intersection with a high bbox.
    `child_weight` is overlap area normalized over all children of its low
    parent. Reverse `parent_weight` is the same overlap area normalized over
    all parents of its high child. The primary parent is the largest overlap;
    ties resolve to the lowest low index.
    """

    low = _coords(low_coords, "low_coords")
    high = _coords(high_coords, "high_coords")
    low_boxes = _make_bboxes(low, low_patch_size, low_downsample)
    high_boxes = _make_bboxes(high, high_patch_size, high_downsample)
    low_valid = _valid_mask(low_boxes)
    high_valid = _valid_mask(high_boxes)
    low_padding = _padding_ratio(low_boxes, wsi_dimensions)
    high_padding = _padding_ratio(high_boxes, wsi_dimensions)

    low_children: list[list[tuple[int, float, float, float]]] = [[] for _ in range(len(low))]
    high_parents: list[list[tuple[int, float]]] = [[] for _ in range(len(high))]
    low_width = float(low_boxes[0, 2] - low_boxes[0, 0]) if len(low_boxes) else 1.0
    low_height = float(low_boxes[0, 3] - low_boxes[0, 1]) if len(low_boxes) else 1.0
    candidates = _candidate_index(high_boxes, low_width, low_height)
    for low_index, low_box in enumerate(low_boxes):
        if not low_valid[low_index]:
            continue
        low_area = (low_box[2] - low_box[0]) * (low_box[3] - low_box[1])
        bx0 = math.floor(low_box[0] / low_width)
        bx1 = math.floor(np.nextafter(low_box[2], -np.inf) / low_width)
        by0 = math.floor(low_box[1] / low_height)
        by1 = math.floor(np.nextafter(low_box[3], -np.inf) / low_height)
        candidate_indices = sorted({
            high_index
            for bx in range(bx0, bx1 + 1)
            for by in range(by0, by1 + 1)
            for high_index in candidates.get((bx, by), ())
        })
        for high_index in candidate_indices:
            high_box = high_boxes[high_index]
            if not high_valid[high_index]:
                continue
            width = min(low_box[2], high_box[2]) - max(low_box[0], high_box[0])
            height = min(low_box[3], high_box[3]) - max(low_box[1], high_box[1])
            if width <= 0.0 or height <= 0.0:
                continue
            area = float(width * height)
            high_area = (high_box[2] - high_box[0]) * (high_box[3] - high_box[1])
            low_children[low_index].append((high_index, area, area / low_area, area / high_area))
            high_parents[high_index].append((low_index, area))

    parent_ptr = [0]
    child_indices: list[int] = []
    child_areas: list[float] = []
    child_weights: list[float] = []
    child_low_fractions: list[float] = []
    child_high_fractions: list[float] = []
    for children in low_children:
        children.sort(key=lambda item: item[0])
        total = sum(item[1] for item in children)
        for high_index, area, low_fraction, high_fraction in children:
            child_indices.append(high_index)
            child_areas.append(area)
            child_weights.append(area / total if total > 0 else 0.0)
            child_low_fractions.append(low_fraction)
            child_high_fractions.append(high_fraction)
        parent_ptr.append(len(child_indices))

    high_parent_ptr = [0]
    parent_indices: list[int] = []
    parent_areas: list[float] = []
    parent_weights: list[float] = []
    primary_parent = np.full(len(high), -1, dtype=np.int64)
    primary_area = np.zeros(len(high), dtype=np.float64)
    for high_index, parents in enumerate(high_parents):
        parents.sort(key=lambda item: item[0])
        total = sum(item[1] for item in parents)
        if parents:
            best = max(parents, key=lambda item: (item[1], -item[0]))
            primary_parent[high_index] = best[0]
            primary_area[high_index] = best[1]
        for low_index, area in parents:
            parent_indices.append(low_index)
            parent_areas.append(area)
            parent_weights.append(area / total if total > 0 else 0.0)
        high_parent_ptr.append(len(parent_indices))

    result = CrossScaleMapping(
        low_bboxes=low_boxes,
        high_bboxes=high_boxes,
        low_valid_mask=low_valid,
        high_valid_mask=high_valid,
        low_padding_ratio=low_padding,
        high_padding_ratio=high_padding,
        parent_ptr=np.asarray(parent_ptr, dtype=np.int64),
        child_indices=np.asarray(child_indices, dtype=np.int64),
        child_overlap_area=np.asarray(child_areas, dtype=np.float64),
        child_weight=np.asarray(child_weights, dtype=np.float64),
        child_low_fraction=np.asarray(child_low_fractions, dtype=np.float64),
        child_high_fraction=np.asarray(child_high_fractions, dtype=np.float64),
        high_parent_ptr=np.asarray(high_parent_ptr, dtype=np.int64),
        parent_indices=np.asarray(parent_indices, dtype=np.int64),
        parent_overlap_area=np.asarray(parent_areas, dtype=np.float64),
        parent_weight=np.asarray(parent_weights, dtype=np.float64),
        unmapped_high_indices=np.flatnonzero(np.diff(np.asarray(high_parent_ptr)) == 0).astype(np.int64),
        high_has_parent_mask=(np.diff(np.asarray(high_parent_ptr)) > 0),
        primary_parent=primary_parent,
        primary_parent_overlap_area=primary_area,
    )
    result.validate()
    return result


__all__ = ["CrossScaleMapping", "build_cross_scale_mapping"]
