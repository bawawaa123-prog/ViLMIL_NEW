import unittest

import torch

from models.cross_scale_modules import HighRouter


def mapping(n_high=4):
    return {
        "high_parent_ptr": torch.tensor([0, 1, 3, 3, 5]),
        "parent_indices": torch.tensor([0, 0, 1, 1, 2]),
        "parent_weight": torch.tensor([1.0, 0.25, 0.75, 0.4, 0.6]),
        "high_has_parent_mask": torch.tensor([True, True, False, True]),
        "unmapped_high_indices": torch.tensor([2]),
        "low_valid_mask": torch.ones(3, dtype=torch.bool),
        "low_padding_ratio": torch.zeros(3),
        "high_valid_mask": torch.ones(n_high, dtype=torch.bool),
        "high_padding_ratio": torch.zeros(n_high),
    }


class SoftRoutingTest(unittest.TestCase):
    def test_zero_init_exact_parity_and_unmapped(self):
        torch.manual_seed(7)
        high = torch.randn(4, 8)
        context = {
            "high_parent_context": torch.randn(4, 8),
            "high_parent_context_valid_mask": torch.tensor([True, True, False, True]),
            "high_valid_mask": torch.ones(4, dtype=torch.bool),
            "high_padding_ratio": torch.zeros(4),
        }
        router = HighRouter(feature_dim=8, hidden_dim=4)
        out = router(high, context)
        torch.testing.assert_close(out, high, rtol=0, atol=0)
        self.assertTrue(torch.equal(out[2], high[2]))

    def test_mapped_gradient_and_finite_empty(self):
        torch.manual_seed(3)
        high = torch.randn(4, 8, requires_grad=True)
        context = {
            "high_parent_context": torch.randn(4, 8, requires_grad=True),
            "high_parent_context_valid_mask": torch.tensor([True, True, False, False]),
            "high_valid_mask": torch.ones(4, dtype=torch.bool),
            "high_padding_ratio": torch.zeros(4),
        }
        router = HighRouter(feature_dim=8, hidden_dim=4)
        # Make a non-zero route update after the parity check.
        with torch.no_grad():
            router.context_projection.weight.fill_(0.05)
        out = router(high, context)
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertGreater(float(router.context_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(context["high_parent_context"].grad[:2].abs().sum()), 0.0)
        self.assertEqual(float(out[2].sub(high.detach()[2]).abs().sum()), 0.0)

    def test_zero_parent_and_padding(self):
        high = torch.randn(2, 8)
        context = {
            "high_parent_context": torch.zeros(2, 8),
            "high_parent_context_valid_mask": torch.tensor([False, True]),
            "high_valid_mask": torch.tensor([True, False]),
            "high_padding_ratio": torch.tensor([0.0, 1.0]),
        }
        out = HighRouter(feature_dim=8, hidden_dim=4)(high, context)
        torch.testing.assert_close(out, high, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
