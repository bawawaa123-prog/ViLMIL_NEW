from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from models.cross_scale_modules import MaskedGatedAttentionPool

def legacy_pool(h,v,u,w):
    return torch.mm(F.softmax(w(v(h)*u(h)).transpose(1,0),dim=1),h)
def err(a,b):
    d=(a-b).abs(); return float(d.max()),float((d/b.abs().clamp_min(1e-12)).max())
def main():
    root=Path(__file__).resolve().parents[3]
    candidates=sorted(p for d in (root/"trained_models",root/"analysis") if d.exists() for p in d.rglob("*") if p.suffix in {".pt",".pth",".ckpt"})
    print("checkpoint_candidates",[str(p) for p in candidates])
    print("checkpoint_strict_load=NOT_RUN (no baseline checkpoint present)" if not candidates else f"checkpoint_strict_load=NOT_RUN (constructor required) path={candidates[0]}")
    torch.manual_seed(3202); d,h,c=32,11,2
    v=nn.Sequential(nn.Linear(d,h),nn.Tanh()); u=nn.Sequential(nn.Linear(d,h),nn.Sigmoid()); w=nn.Linear(h,1); pool=MaskedGatedAttentionPool()
    lo,hi=torch.randn(7,d),torch.randn(19,d); tl,th=torch.randn(c,d),torch.randn(c,d)
    llo,lhi=legacy_pool(lo,v,u,w)@tl.T,legacy_pool(hi,v,u,w)@th.T
    rlo,rhi=pool(lo,v,u,w)[0]@tl.T,pool(hi,v,u,w)[0]@th.T
    for mode,a,b in (("high",rhi,lhi),("low",rlo,llo),("dual",rlo+rhi,llo+lhi)):
        ae,re=err(a,b); print(f"{mode}_logits_max_abs_error={ae:.9g} {mode}_logits_max_rel_error={re:.9g}"); torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    print("scale_mode_semantics=PASS")
if __name__ == "__main__": main()
