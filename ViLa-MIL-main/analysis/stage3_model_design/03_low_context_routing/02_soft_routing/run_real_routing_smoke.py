"""Small real-feature Stage 3.3.2 routing smoke (one slide, no training)."""
from types import SimpleNamespace
from pathlib import Path
import sys
import time

import h5py
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
from utils.cross_scale_mapping import CrossScaleMapping


SLIDE = "2460239-B2"
MAP = ROOT / "analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings" / f"{SLIDE}.npz"
LOW = ROOT / "data/yiyuan/features_biomedclip_5x" / f"{SLIDE}.h5"
HIGH = ROOT / "data/yiyuan/features_biomedclip_20x" / f"{SLIDE}.h5"


def main():
    with h5py.File(LOW, "r") as handle:
        low = torch.from_numpy(handle["features"][:]).unsqueeze(0).float()
        low_coord = torch.from_numpy(handle["coords"][:]).unsqueeze(0)
    with h5py.File(HIGH, "r") as handle:
        high = torch.from_numpy(handle["features"][:]).unsqueeze(0).float()
        high_coord = torch.from_numpy(handle["coords"][:]).unsqueeze(0)
    mapping = CrossScaleMapping.load_npz(MAP).to_dict()
    config = SimpleNamespace(hidden_size=192, text_prompt=["low", "low", "high", "high"],
                             prototype_number=2, scale_mode="high",
                             use_low_context_routing=True, finetune_text_encoder=False,
                             text_finetune_mode="proj", text_unfreeze_last_n=2)
    torch.manual_seed(11)
    model = ViLa_MIL_BiomedCLIP(config=config, num_classes=2)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        prob, pred, loss = model(low, low_coord, high, high_coord, torch.tensor([0]), mapping=mapping)
    elapsed = time.perf_counter() - start
    assert torch.isfinite(prob).all() and torch.isfinite(loss)
    assert model._last_mapping_context["high_parent_context"].shape[0] == high.shape[1]
    context = model._last_mapping_context["high_parent_context"]
    context_mb = context.numel() * context.element_size() / (1024 ** 2)
    print(f"{SLIDE}: low={low.shape[1]} high={high.shape[1]} prob_shape={tuple(prob.shape)} finite=True elapsed_s={elapsed:.3f} context_mb={context_mb:.2f}")


if __name__ == "__main__":
    main()
