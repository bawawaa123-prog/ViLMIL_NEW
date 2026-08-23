from __future__ import annotations
import sys
from pathlib import Path
import h5py
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.cross_scale_modules import LowParentContext
from utils.cross_scale_mapping import CrossScaleMapping

def main():
    slide = "2460239-B2"
    cache = ROOT / "analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings" / f"{slide}.npz"
    with h5py.File(ROOT / "data/yiyuan/features_biomedclip_5x" / f"{slide}.h5", "r") as h:
        low = torch.from_numpy(h["features"][:]).float()
    with h5py.File(ROOT / "data/yiyuan/features_biomedclip_20x" / f"{slide}.h5", "r") as h:
        high_count = len(h["features"])
    m = CrossScaleMapping.load_npz(str(cache)).to_dict()
    out = LowParentContext()(low, m, high_count=high_count)
    assert out["high_parent_context"].shape == (high_count, low.shape[1])
    assert torch.isfinite(out["high_parent_context"]).all()
    assert int((~out["high_has_parent_mask"]).sum()) == len(m["unmapped_high_indices"])
    print(f"{slide}: low={len(low)} high={high_count} unmapped={len(m['unmapped_high_indices'])} context_shape={tuple(out['high_parent_context'].shape)}")

if __name__ == "__main__":
    main()
