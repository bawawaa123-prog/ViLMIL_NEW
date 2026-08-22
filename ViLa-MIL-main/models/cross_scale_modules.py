"""Reusable parameter-free pooling primitives for Stage 3.2.

The attention projections remain owned by the legacy model so their state_dict
paths (`attention_V.*`, `attention_U.*`, `attention_weights.*`) do not change.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedGatedAttentionPool(nn.Module):
    """Legacy gated-attention pooling with optional invalid-row masking.

    For a non-empty all-valid input this is exactly:
    ``softmax(attention_weights(attention_V(H) * attention_U(H)).T) @ H``.
    The module owns no trainable parameters. Empty or fully invalid inputs return
    a finite zero vector and zero attention weights.
    """

    def forward(
        self,
        features: torch.Tensor,
        attention_V: nn.Module,
        attention_U: nn.Module,
        attention_weights: nn.Module,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
        n_rows = features.shape[0]
        if mask is None:
            valid = torch.ones(n_rows, dtype=torch.bool, device=features.device)
        else:
            valid = torch.as_tensor(mask, device=features.device, dtype=torch.bool)
            if valid.ndim != 1 or valid.shape[0] != n_rows:
                raise ValueError(f"mask must be [N]={n_rows}, got {tuple(valid.shape)}")

        # Keep the legacy operation order and dtype for parity.
        hidden = features.float()
        gated = attention_V(hidden) * attention_U(hidden)
        scores = attention_weights(gated).transpose(1, 0)
        masked_scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        if bool(valid.any()):
            weights = F.softmax(masked_scores, dim=1)
            pooled = torch.mm(weights, hidden)
            weights = weights * valid.unsqueeze(0).to(weights.dtype)
        else:
            weights = torch.zeros_like(scores)
            pooled = hidden.new_zeros((scores.shape[0], hidden.shape[1]))
        return pooled, weights


__all__ = ["MaskedGatedAttentionPool"]
