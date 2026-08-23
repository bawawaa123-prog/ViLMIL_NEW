from __future__ import annotations

import unittest
import torch

from models.cross_scale_modules import LowParentContext


def mapping(n_low=4, n_high=5):
    return {
        "high_parent_ptr": [0, 0, 1, 3, 3, 3],
        "parent_indices": [1, 0, 2],
        "parent_weight": [1.0, 0.25, 0.75],
        "low_valid_mask": [True, True, True, True],
        "high_valid_mask": [True] * n_high,
        "low_padding_ratio": [0.0] * n_low,
        "high_padding_ratio": [0.0] * n_high,
        "high_has_parent_mask": [False, True, True, False, False],
        "unmapped_high_indices": [0, 3, 4],
    }


class MappingContextTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3311)
        self.low = torch.tensor([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]])
        self.module = LowParentContext()

    def test_zero_one_many_and_unmapped(self):
        out = self.module(self.low, mapping())
        torch.testing.assert_close(out["high_parent_context"][0], torch.zeros(2))
        torch.testing.assert_close(out["high_parent_context"][1], self.low[1])
        torch.testing.assert_close(out["high_parent_context"][2], self.low[0] * .25 + self.low[2] * .75)
        self.assertEqual(out["unmapped_high_indices"].tolist(), [0, 3, 4])
        self.assertEqual(out["high_parent_context_valid_mask"].tolist(), [False, True, True, False, False])

    def test_invalid_low_and_padding_are_explicit(self):
        m = mapping()
        m["low_valid_mask"] = [False, True, True, True]
        m["low_padding_ratio"] = [1.0, 0.0, 0.0, 0.0]
        out = self.module(self.low, m)
        torch.testing.assert_close(out["high_parent_context"][2], self.low[2])
        self.assertTrue(torch.isfinite(out["high_parent_context"]).all())
        self.assertEqual(out["high_padding_ratio"].shape[0], 5)

    def test_empty_segments_and_backward_finite(self):
        low = self.low.clone().requires_grad_(True)
        out = self.module(low, mapping())
        loss = out["high_parent_context"].square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(out["high_parent_context"]).all())
        self.assertTrue(torch.isfinite(low.grad).all())

    def test_row_alignment_and_bad_indices(self):
        out = self.module(self.low, mapping())
        self.assertEqual(out["high_parent_context"].shape, (5, 2))
        bad = mapping(); bad["parent_indices"] = [99, 0, 2]
        with self.assertRaises(IndexError):
            self.module(self.low, bad)

if __name__ == "__main__":
    unittest.main(verbosity=2)
