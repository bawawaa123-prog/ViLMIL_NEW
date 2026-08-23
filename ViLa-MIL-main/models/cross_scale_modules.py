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


class LowParentContext(nn.Module):
    """Parameter-free reverse-CSR low context for each high feature row.

    ``parent_indices`` and ``parent_weight`` index rows of ``low_features``;
    therefore this module never uses standalone coordinate-H5 row order. Invalid
    low parents are removed and the remaining weights are renormalized. High
    rows with no usable parent receive a finite zero fallback and remain present.
    """

    def forward(
        self,
        low_features: torch.Tensor,
        mapping: dict,
        *,
        high_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if low_features.ndim != 2:
            raise ValueError(f"low_features must be [N_low, D], got {tuple(low_features.shape)}")
        device = low_features.device
        dtype = low_features.dtype
        n_low, dim = low_features.shape
        def tensor(name, dtype=None):
            if name not in mapping:
                raise KeyError(f"mapping is missing {name!r}")
            return torch.as_tensor(mapping[name], device=device, dtype=dtype)

        ptr = tensor("high_parent_ptr", torch.long)
        parents = tensor("parent_indices", torch.long)
        weights = tensor("parent_weight", dtype)
        n_high = int(ptr.numel() - 1) if high_count is None else int(high_count)
        if ptr.numel() != n_high + 1 or ptr[0].item() != 0 or ptr[-1].item() != parents.numel():
            raise ValueError("reverse CSR pointer/index sizes are inconsistent")
        if parents.numel() and (parents.min().item() < 0 or parents.max().item() >= n_low):
            raise IndexError("parent_indices do not index low feature rows")
        if weights.numel() != parents.numel():
            raise ValueError("parent_weight and parent_indices sizes differ")

        low_valid = tensor("low_valid_mask", torch.bool)
        if low_valid.numel() != n_low:
            raise ValueError("low_valid_mask does not match low feature rows")
        low_padding = tensor("low_padding_ratio", dtype)
        if low_padding.numel() != n_low:
            raise ValueError("low_padding_ratio does not match low feature rows")
        high_valid = tensor("high_valid_mask", torch.bool)
        high_padding = tensor("high_padding_ratio", dtype)
        if high_valid.numel() != n_high or high_padding.numel() != n_high:
            raise ValueError("high masks do not match high feature rows")

        context = low_features.new_zeros((n_high, dim))
        context_valid = torch.zeros(n_high, dtype=torch.bool, device=device)
        for high_index in range(n_high):
            start, end = int(ptr[high_index]), int(ptr[high_index + 1])
            if end <= start or not bool(high_valid[high_index]):
                continue
            p = parents[start:end]
            w = weights[start:end]
            usable = low_valid[p] & torch.isfinite(w) & (w > 0)
            if not bool(usable.any()):
                continue
            p, w = p[usable], w[usable]
            w = w / w.sum().clamp_min(torch.finfo(dtype).eps)
            context[high_index] = torch.sum(low_features[p] * w.unsqueeze(1), dim=0)
            context_valid[high_index] = True

        has_parent = tensor("high_has_parent_mask", torch.bool)
        unmapped = tensor("unmapped_high_indices", torch.long)
        if has_parent.numel() != n_high:
            raise ValueError("high_has_parent_mask does not match high rows")
        if not torch.equal(unmapped, torch.where(~has_parent)[0]):
            raise ValueError("unmapped_high_indices disagrees with high_has_parent_mask")
        return {
            "high_parent_context": context,
            "high_parent_context_valid_mask": context_valid,
            "high_has_parent_mask": has_parent,
            "unmapped_high_indices": unmapped,
            "high_valid_mask": high_valid,
            "high_padding_ratio": high_padding,
        }


__all__.append("LowParentContext")


class HighRouter(nn.Module):
    """Soft, residual low-conditioned routing for raw high features.

    The context projection is zero initialized, so enabling the module starts
    exactly at the legacy high path.  Route scores are bounded and metadata
    gates ensure rows without a usable low parent remain unchanged.
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.context_projection = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)
        self.route_score = nn.Sequential(
            nn.Linear(feature_dim * 2 + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # A zero route logit gives a neutral 0.5 soft gate while the residual
        # itself is zero at initialization; this keeps exact feature parity.
        nn.init.zeros_(self.route_score[-1].weight)
        nn.init.zeros_(self.route_score[-1].bias)

    def forward(self, high_features: torch.Tensor, context: dict[str, torch.Tensor]) -> torch.Tensor:
        if high_features.ndim != 2:
            raise ValueError(f"high_features must be [N, D], got {tuple(high_features.shape)}")
        low_context = context["high_parent_context"].to(device=high_features.device, dtype=high_features.dtype)
        if low_context.shape != high_features.shape:
            raise ValueError(f"context shape {tuple(low_context.shape)} != high shape {tuple(high_features.shape)}")
        valid = context["high_parent_context_valid_mask"].to(high_features.device, dtype=torch.bool)
        high_valid = context.get("high_valid_mask")
        if high_valid is not None:
            valid = valid & high_valid.to(high_features.device, dtype=torch.bool)
        padding = context.get("high_padding_ratio")
        if padding is None:
            padding = torch.zeros(high_features.shape[0], device=high_features.device, dtype=high_features.dtype)
        else:
            padding = padding.to(high_features.device, dtype=high_features.dtype)
        metadata = torch.stack([valid.to(high_features.dtype), padding], dim=1)
        route_input = torch.cat([high_features, low_context, metadata], dim=1)
        route = torch.sigmoid(self.route_score(route_input))
        route = route * valid.to(route.dtype).unsqueeze(1)
        return high_features + route * self.context_projection(low_context)


__all__.append("HighRouter")
