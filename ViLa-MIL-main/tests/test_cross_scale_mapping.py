import numpy as np
import unittest
import tempfile

from utils.cross_scale_mapping import build_cross_scale_mapping


KW = dict(low_patch_size=10, high_patch_size=4, low_downsample=1, high_downsample=1)


class CrossScaleMappingTests(unittest.TestCase):
  def test_variable_children_and_unmapped_high(self):
    result = build_cross_scale_mapping(
        np.array([[0, 0], [20, 0], [40, 0]]),
        np.array([[1, 1], [5, 1], [21, 1], [100, 100]]),
        **KW,
    )
    assert result.low_child_counts.tolist() == [2, 1, 0]
    assert result.unmapped_high_indices.tolist() == [3]
    assert result.high_has_parent_mask.tolist() == [True, True, True, False]


  def test_partial_overlap_and_multiple_parents(self):
    result = build_cross_scale_mapping(
        np.array([[0, 0], [8, 0]]),
        np.array([[6, 1]]),
        low_patch_size=10,
        high_patch_size=6,
        low_downsample=1,
        high_downsample=1,
    )
    assert result.parents(0).tolist() == [0, 1]
    assert np.allclose(result.parent_overlap_area, [24.0, 24.0])
    assert np.allclose(result.parent_weight, [0.5, 0.5])
    assert result.primary_parent.tolist() == [0]


  def test_non_integer_scale_and_bbox_semantics(self):
    result = build_cross_scale_mapping(
        np.array([[0.25, 0.75]]),
        np.array([[4.5, 4.9]]),
        low_patch_size=4,
        high_patch_size=2,
        low_downsample=2.5,
        high_downsample=1.01,
    )
    assert result.child_indices.tolist() == [0]
    assert result.child_overlap_area[0] > 0
    assert result.low_bboxes[0].tolist() == [0.25, 0.75, 10.25, 10.75]


  def test_csr_integrity_weights_masks_and_determinism(self):
    low = np.array([[0, 0], [0, 10]])
    high = np.array([[1, 1], [0, 11], [50, 50]])
    first = build_cross_scale_mapping(low, high, **KW)
    second = build_cross_scale_mapping(low, high, **KW)
    assert np.array_equal(first.parent_ptr, second.parent_ptr)
    assert np.array_equal(first.child_indices, second.child_indices)
    assert first.parent_ptr.tolist() == [0, 1, 2]
    assert first.high_parent_ptr.tolist() == [0, 1, 2, 2]
    assert len(first.child_indices) == int(first.parent_ptr[-1])
    for i in range(len(low)):
        start, end = first.parent_ptr[i : i + 2]
        assert np.isclose(first.child_weight[start:end].sum(), 1.0)
    assert first.low_valid_mask.shape == (2,)
    assert first.high_valid_mask.shape == (3,)
    assert first.unmapped_high_indices.size == 1


  def test_padding_ratio_and_invalid_geometry_are_explicit(self):
    result = build_cross_scale_mapping(
        np.array([[-2, -2], [0, 0]]),
        np.array([[-1, -1], [1, 1]]),
        low_patch_size=4,
        high_patch_size=2,
        low_downsample=1,
        high_downsample=1,
        wsi_dimensions=(4, 4),
    )
    assert result.low_padding_ratio[0] > 0
    assert result.high_padding_ratio[0] > 0
    assert result.low_valid_mask.all() and result.high_valid_mask.all()

  def test_npz_round_trip_preserves_mapping(self):
    result = build_cross_scale_mapping(
        np.array([[0, 0]]), np.array([[1, 1], [30, 30]]), **KW
    )
    with tempfile.TemporaryDirectory() as directory:
      path = f"{directory}/mapping.npz"
      result.save_npz(path)
      restored = result.load_npz(path)
    assert np.array_equal(restored.parent_ptr, result.parent_ptr)
    assert np.array_equal(restored.child_indices, result.child_indices)
    assert np.array_equal(restored.child_weight, result.child_weight)
    assert np.array_equal(restored.unmapped_high_indices, result.unmapped_high_indices)


if __name__ == "__main__":
    unittest.main()
