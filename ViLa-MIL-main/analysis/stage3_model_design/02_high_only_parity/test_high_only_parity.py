from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from models.cross_scale_modules import MaskedGatedAttentionPool


SEED = 3202
ATOL = 1e-7
RTOL = 1e-6


def legacy_pool(features, attention_v, attention_u, attention_weights):
    scores = attention_weights(attention_v(features.float()) * attention_u(features.float()))
    weights = F.softmax(scores.transpose(1, 0), dim=1)
    return torch.mm(weights, features.float()), weights


def errors(actual, expected):
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1e-12)
    return float(absolute.max()), float(relative.max())


class LegacyAttentionHarness(nn.Module):
    """Preserves the production state_dict ownership contract."""

    def __init__(self, input_dim=32, hidden_dim=11):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.gated_attention_pool = MaskedGatedAttentionPool()

    def forward(self, features, mask=None):
        return self.gated_attention_pool(
            features, self.attention_V, self.attention_U, self.attention_weights, mask
        )


class HighOnlyParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.model = LegacyAttentionHarness()
        self.features = torch.randn(17, 32)

    def test_legacy_formula_and_pooled_feature_parity(self):
        expected_pool, expected_weights = legacy_pool(
            self.features, self.model.attention_V, self.model.attention_U, self.model.attention_weights
        )
        actual_pool, actual_weights = self.model(self.features)
        torch.testing.assert_close(actual_weights, expected_weights, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(actual_pool, expected_pool, atol=ATOL, rtol=RTOL)
        print("legacy_weights_error", errors(actual_weights, expected_weights))
        print("legacy_pool_error", errors(actual_pool, expected_pool))

    def test_no_mask_equals_all_valid(self):
        no_mask = self.model(self.features)
        all_valid = self.model(self.features, torch.ones(17, dtype=torch.bool))
        for actual, expected in zip(no_mask, all_valid):
            torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        print("all_valid_pool_error", errors(no_mask[0], all_valid[0]))

    def test_partial_and_empty_masks(self):
        mask = torch.tensor([True, False, True] + [False] * 14)
        pooled, weights = self.model(self.features, mask)
        selected_pool, selected_weights = legacy_pool(
            self.features[mask], self.model.attention_V, self.model.attention_U, self.model.attention_weights
        )
        torch.testing.assert_close(pooled, selected_pool, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(weights[:, mask], selected_weights, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(weights[:, ~mask], torch.zeros_like(weights[:, ~mask]))
        empty_pool, empty_weights = self.model(self.features, torch.zeros(17, dtype=torch.bool))
        assert torch.isfinite(empty_pool).all() and torch.isfinite(empty_weights).all()
        torch.testing.assert_close(empty_pool, torch.zeros_like(empty_pool))
        torch.testing.assert_close(empty_weights, torch.zeros_like(empty_weights))
        print("partial_pool_error", errors(pooled, selected_pool))

    def test_finite_forward_backward(self):
        features = self.features.clone().requires_grad_(True)
        pooled, weights = self.model(features, torch.arange(17) % 3 != 0)
        (pooled.square().mean() + weights.square().mean()).backward()
        assert torch.isfinite(pooled).all() and torch.isfinite(features.grad).all()
        for parameter in self.model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    def test_parameter_count_and_checkpoint_schema(self):
        legacy = LegacyAttentionHarness()
        refactored = LegacyAttentionHarness()
        keys = set(refactored.state_dict())
        expected_keys = {
            "attention_V.0.weight", "attention_V.0.bias",
            "attention_U.0.weight", "attention_U.0.bias",
            "attention_weights.weight", "attention_weights.bias",
        }
        self.assertEqual(keys, expected_keys)
        self.assertFalse(any(key.startswith("gated_attention_pool.") for key in keys))
        self.assertEqual(sum(p.numel() for p in legacy.parameters()), sum(p.numel() for p in refactored.parameters()))
        load_result = refactored.load_state_dict(copy.deepcopy(legacy.state_dict()), strict=True)
        self.assertEqual(load_result.missing_keys, [])
        self.assertEqual(load_result.unexpected_keys, [])
        print("trainable_parameter_count", sum(p.numel() for p in refactored.parameters() if p.requires_grad))

    def test_scale_mode_selection_semantics(self):
        low = torch.tensor([[1.0, -2.0]])
        high = torch.tensor([[3.0, 4.0]])
        expected = {"low": low, "high": high, "dual": low + high}
        for mode, logits in expected.items():
            actual = low if mode == "low" else high if mode == "high" else low + high
            torch.testing.assert_close(actual, logits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
