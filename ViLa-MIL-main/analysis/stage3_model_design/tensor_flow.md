# Tensor Flow Contract

The current loader returns one slide per batch with a leading singleton dimension. The proposed implementation may normalize this to `[N, d]` internally and restore `[1, ...]` only at the public boundary.

## Inputs

| Tensor | Shape | Meaning |
|---|---:|---|
| `x_low` | `[1, N_s, 512]` | actual ~2.5x features |
| `coord_low` | `[1, N_s, 2]` | feature-H5 coordinates, level-0 top-left |
| `x_high` | `[1, N_h, 512]` | actual ~10x features |
| `coord_high` | `[1, N_h, 2]` | feature-H5 coordinates, level-0 top-left |
| `mapping` | ragged or CSR | `low -> high` variable children, overlap weights |
| `high_parent_mask` | `[N_h] bool` | true if at least one low parent |
| `padding_low/high` | `[N_s]`, `[N_h] float/bool` | boundary/padding metadata |
| `label` | `[1]` | slide class |

The mapper may expose CSR tensors `child_index [E]`, `child_weight [E]`, `parent_ptr [N_s+1]`, plus `unmapped_high_index [N_u]`. `E` is not constrained to `16*N_s`.

## MVP forward

1. `h_s = LN(MLP_s(x_low))`, `h_l = LN(MLP_h(x_high))`; both `[N, d]`, with `d = config.hidden_size` (for example 256).
2. Low region attention: `r_s = MaskedGatedPool(h_s, padding_low)` -> `[N_s, d]`.
3. For each CSR segment, `c_i = sum_j w_ij r_l[j]` over high children; missing parent yields learned `c_unmapped` plus `unmapped_mask`.
4. Text prompts produce normalized `T_high [C, 512]`; project to `T [C, d]` or compute compatibility in 512-D. `semantic_i = max_c cos(h_l_i, T_c)` or class-specific vector `[N_h, C]`.
5. `route_i = sigmoid(MLP([h_l_i, c_i, semantic_i, overlap_stats_i, padding_i, unmapped_mask_i]))` -> `[N_h, 1]`.
6. High evidence pool: `z_high = sum_i softmax(masked_score_i) * (h_l_i + W_c c_i * route_i)` -> `[1, d]`.
7. Low slide context: `z_low = MaskedGatedPool(r_s)` -> `[1, d]`.
8. High classifier: `logits_high = W_o z_high` -> `[1, C]`.
9. Asymmetric residual: `g = sigmoid(W_g[z_low; z_high])`, `r = W_r z_low`; `logits = logits_high + alpha * g * W_delta[r; z_high]`, with `alpha` initialized to 0.1. The residual is zero/near-zero at initialization and cannot force equal-scale voting.

## Loss

MVP objective is slide cross-entropy only:

`L = CE(logits, label)`.

Do not add low auxiliary CE in the first causal experiment because Low-only is weaker and an auxiliary classifier would reintroduce pressure for low to act as a peer diagnosis branch. Optional later terms are `lambda_route * entropy/coverage regularizer` and `lambda_aux * CE(logits_high, label)`, each gated by an ablation.

## Required invariants

- `sum(parent_ptr[1:] - parent_ptr[:-1]) == E`; no fixed cardinality assertion.
- Every high row appears either in at least one mapping segment or in `unmapped_high_index`.
- Padding and unmapped masks alter attention logits, never feature-row alignment.
- All reductions are finite for `N_s=0`, `N_h=0`, and empty child segments via explicit fallback vectors.

