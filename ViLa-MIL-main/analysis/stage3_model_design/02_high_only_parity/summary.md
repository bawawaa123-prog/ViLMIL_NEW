# Stage 3.2 High-only Parity Summary

## Scope

Stage 3.2 extracts the existing gated-attention plus weighted pooling formula into `models/cross_scale_modules.py`. It does not consume Stage 3.1 mapping, add routing, change classifiers/losses, or add diagnostic capability. Both production calls use `mask=None`; masks are only exercised by isolated tests.

## Implementation

- Added `MaskedGatedAttentionPool`, a parameter-free `nn.Module` accepting `[N, D]` features, the existing `attention_V/U/weights` modules, and optional `[N]` boolean mask.
- For non-empty all-valid input, operation order is unchanged: `attention_V(H) * attention_U(H)`, `attention_weights`, transpose, softmax over patches, matrix multiply with `H`.
- Partial masks set invalid attention entries to `-inf`; invalid rows receive zero weight and do not enter the pooled feature. Empty/all-invalid masks return finite zero pooled features and zero weights.
- Updated both `forward` and `forward_with_attention` in `models/model_ViLa_MIL_BiomedCLIP.py` to call the reusable module with `mask=None`. Low, high, and dual scale selection logic remains unchanged.
- The wrapper owns no parameters. Existing modules remain registered at the model root, so `attention_V.*`, `attention_U.*`, and `attention_weights.*` state_dict keys are unchanged. Prototype decoder, image center, both cross-attention modules, BiomedCLIP, prompt learner, and text encoder remain intact.

## Tests

Command:

```bash
python -m unittest discover \
  -s analysis/stage3_model_design/02_high_only_parity \
  -p 'test_*.py' -v
```

Result: 6 tests passed in 0.316 seconds with fixed seed `3202`.

| Check | Result | Error/details |
|---|---|---|
| Legacy formula vs new module | pass | max abs/rel error `0.0 / 0.0` for weights and pooled feature |
| `mask=None` vs all-valid | pass | max error `0.0` |
| Partial mask | pass | max abs `5.96e-08`, max rel `2.44e-07` |
| Empty mask finite behavior | pass | finite zero pooled feature and weights |
| Forward/backward | pass | finite outputs, inputs gradients, and parameter gradients |
| Parameter count | pass | trainable parameter count `738` in fixed harness; wrapper adds zero |
| State dict compatibility | pass | strict load has no missing/unexpected keys; no `gated_attention_pool.*` keys |
| Scale mode semantics | pass | low/high/dual selection unchanged |

Syntax and whitespace checks also passed:

```bash
python -m py_compile \
  models/cross_scale_modules.py \
  models/model_ViLa_MIL_BiomedCLIP.py \
  analysis/stage3_model_design/02_high_only_parity/test_high_only_parity.py
git diff --check
```

## Compatibility decision

No feature, coordinate, mapping cache, dataset contract, checkpoint file, or training configuration was changed. The production model still receives `(x_s, coord_s, x_l, coords_l, label)`, and `scale_mode=low/high/dual` remains available. Stage 3.2 is accepted and the code is ready for Stage 3.3; do not begin Stage 3.3 in this change.

## Final acceptance additions

Existing baseline checkpoints are present. The selected representative is `analysis/stage2_yiyuan_data_audit/06_scale_ablation/high_only/adenocarcinoma_biomedclip_high_only_strict5_s1/s_0_checkpoint.pt`; it is an `OrderedDict` with 372 keys, including unchanged `attention_V.*`, `attention_U.*`, and `attention_weights.*`. A real model-constructor `strict=True` load was not run in this short acceptance turn because it requires BiomedCLIP initialization/cache. Therefore no missing/unexpected-key result is claimed. The acceptance script reports the selected path explicitly; run it after the local BiomedCLIP cache is ready.

The complete post-prototype low/high/dual logits path was parity-tested with fixed seed `3202`, identical parameters and inputs, and `torch.testing.assert_close(atol=rtol=1e-6)`. High logits, low logits, and dual logits all pass; the script prints max absolute and relative errors.

Command: `python analysis/stage3_model_design/02_high_only_parity/final_acceptance.py`

This short smoke test does not train or initialize/download BiomedCLIP.
