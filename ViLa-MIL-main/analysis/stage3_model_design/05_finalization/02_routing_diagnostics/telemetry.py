"""Read-only hooks; no modified forward, parameters, buffers or returned tensors."""
import numpy as np
import torch
from bridge import GATE_LOW, GATE_HIGH, EPS


def distribution(values, prefix):
    values = np.asarray(values, dtype=float)
    return {f'{prefix}_{k}': float(fn(values)) if len(values) else float('nan')
            for k, fn in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max),
                          ('p05', lambda x: np.quantile(x,.05)), ('median', np.median),
                          ('p95', lambda x: np.quantile(x,.95))]}


def parent_statistics(gates, valid, mapping):
    """Overlapping groups: every valid edge retained, weighted within each Low parent."""
    ptr = np.asarray(mapping['parent_ptr'])
    children = np.asarray(mapping['child_indices'])
    weights = np.asarray(mapping['child_weight'])
    low_valid = np.asarray(mapping['low_valid_mask'], dtype=bool)
    means, variances, counts = [], [], []
    for parent, (start,end) in enumerate(zip(ptr[:-1],ptr[1:])):
        if not low_valid[parent]:
            continue
        ids, w = children[start:end], weights[start:end]
        usable = valid[ids] & np.isfinite(w) & (w > 0)
        ids, w = ids[usable], w[usable]
        if not len(ids):
            continue
        w = w / w.sum()
        mean = float(np.sum(w*gates[ids]))
        means.append(mean)
        counts.append(len(ids))
        variances.append(float(np.sum(w*(gates[ids]-mean)**2)))
    within = [v for v,n in zip(variances,counts) if n>1]
    result = dict(low_parent_groups=len(means), multipatch_parent_groups=len(within),
                  within_parent_gate_variance_mean=float(np.mean(within)) if within else np.nan,
                  between_parent_gate_mean_variance=float(np.var(means)) if len(means)>1 else np.nan)
    regions = [dict(low_parent=p, gate_mean=m, gate_variance=v, n_children=n)
               for p,(m,v,n) in enumerate(zip(means,variances,counts))]
    # The returned group IDs must be original Low feature row IDs, never enumeration after filtering.
    original = [p for p,(start,end) in enumerate(zip(ptr[:-1],ptr[1:]))
                if low_valid[p] and np.any(valid[children[start:end]] & np.isfinite(weights[start:end]) & (weights[start:end]>0))]
    for row,p in zip(regions,original):
        row['low_parent'] = int(p)
    return result, regions


class RouterRecorder:
    def __init__(self, router):
        self.handles = [router.route_score.register_forward_hook(self._score),
                        router.context_projection.register_forward_hook(self._projection),
                        router.register_forward_hook(self._router)]
        self.row = None
        self.patches = []
        self.regions = []
        self.histograms = []
        self.mapping = None
        self.coords = None
        self.seed = 0
        self.sample_limit = 256

    def _score(self, module, args, output):
        self.score = output.detach()

    def _projection(self, module, args, output):
        self.projected = output.detach()

    def _router(self, module, args, output):
        high, context = args[:2]
        valid = context['high_parent_context_valid_mask'].bool()
        if 'high_valid_mask' in context:
            valid = valid & context['high_valid_mask'].bool()
        gates = torch.sigmoid(self.score).reshape(-1) * valid.to(high.dtype)
        # Required modes leave gates unchanged; residual_off sets residual to zero before capping.
        mode = module.last_diagnostics['diagnostic_mode']
        if mode not in ('normal','residual_off'):
            raise ValueError('Recorder supports only declared A0-A3 intervention gates')
        raw = gates[:,None]*self.projected
        if mode == 'residual_off':
            raw = torch.zeros_like(raw)
        hn = high.norm(dim=1)
        raw_norm = raw.norm(dim=1)
        cap = raw_norm > module.residual_ratio*hn if module.stabilize else torch.zeros_like(valid)
        scale = torch.minimum(torch.ones_like(hn), module.residual_ratio*hn/raw_norm.clamp_min(EPS)) if module.stabilize else torch.ones_like(hn)
        residual = raw*scale[:,None]
        # Observe actual forwarded effect separately: subtraction has float32 cancellation error.
        observed = (output.detach()-high.detach()).norm(dim=1)
        rn = residual.norm(dim=1)
        arrays = {k:v.detach().cpu().numpy() for k,v in
                  [('gate',gates),('valid',valid),('high_norm',hn),('raw_residual_norm',raw_norm),
                   ('residual_norm',rn),('ratio',rn/hn.clamp_min(EPS)),('cap',cap),('observed_norm',observed)]}
        valid_np = arrays['valid'].astype(bool)
        mapped_gates = arrays['gate'][valid_np]
        mapped_ratios = arrays['ratio'][valid_np]
        if np.any(arrays['ratio'] > module.residual_ratio+1e-6) and module.stabilize:
            raise ValueError('Per-patch residual exceeds frozen cap')
        if np.any(arrays['residual_norm'][~valid_np] != 0):
            raise ValueError('Invalid/unmapped rows acquired nonzero residual')
        mapping_has_parent = np.asarray(self.mapping['high_has_parent_mask'], dtype=bool)
        gate_stats = {**distribution(mapped_gates,'gate'),
                      'gate_variance':float(np.var(mapped_gates)) if len(mapped_gates) else np.nan,
                      'gate_near_zero_fraction':float((mapped_gates<GATE_LOW).mean()) if len(mapped_gates) else np.nan,
                      'gate_near_one_fraction':float((mapped_gates>GATE_HIGH).mean()) if len(mapped_gates) else np.nan}
        groups, self.regions = parent_statistics(arrays['gate'],valid_np,self.mapping)
        high_total = float(np.linalg.norm(arrays['high_norm']))
        self.row = {**gate_stats, **groups, **distribution(mapped_ratios,'mapped_residual_ratio'),
                    **distribution(arrays['high_norm'],'patch_high_norm'),
                    **distribution(arrays['raw_residual_norm'][valid_np],'mapped_raw_residual_norm'),
                    **distribution(arrays['residual_norm'][valid_np],'mapped_residual_norm'),
                    'n_high':len(valid_np), 'n_effective_mapped':int(valid_np.sum()),
                    'n_mapping_unmapped':int((~mapping_has_parent).sum()),
                    'mapping_unmapped_fraction':float((~mapping_has_parent).mean()),
                    'n_no_usable_context':int((~valid_np).sum()),
                    'no_usable_context_fraction':float((~valid_np).mean()),
                    'high_frobenius_norm':high_total,
                    'residual_frobenius_norm':float(np.linalg.norm(arrays['residual_norm'])),
                    'aggregate_residual_high_ratio':float(np.linalg.norm(arrays['residual_norm'])/max(high_total,EPS)),
                    'cap_trigger_fraction_all':float(arrays['cap'].mean()),
                    'cap_trigger_fraction_mapped':float(arrays['cap'][valid_np].mean()) if valid_np.any() else np.nan,
                    'observed_residual_frobenius_norm':float(np.linalg.norm(arrays['observed_norm']))}
        for name,values,edges in [('gate',mapped_gates,np.linspace(0,1,21)),
                                  ('residual_ratio',mapped_ratios,np.linspace(0,.10,21))]:
            # Norm rounding may produce rho+1 ulp; values were checked against rho above.
            values = np.clip(values,edges[0],edges[-1])
            counts,_ = np.histogram(values,bins=edges)
            if counts.sum() != len(values):
                raise ValueError('Histogram dropped diagnostic values')
            self.histograms.extend(dict(quantity=name,bin_left=float(edges[i]),bin_right=float(edges[i+1]),count=int(n))
                                   for i,n in enumerate(counts))
        rng = np.random.default_rng(self.seed)
        ids = np.sort(rng.choice(len(valid_np),size=min(self.sample_limit,len(valid_np)),replace=False))
        coords = np.asarray(self.coords)
        self.patches = [dict(patch_index=int(i), x=float(coords[i,0]),y=float(coords[i,1]),
                             **{k:float(v[i]) for k,v in arrays.items()}) for i in ids]

    def prepare(self, mapping, coords, seed, limit=256):
        self.mapping, self.coords, self.seed, self.sample_limit = mapping, coords, seed, limit
        self.row, self.patches, self.regions, self.histograms = None, [], [], []

    def close(self):
        for handle in self.handles:
            handle.remove()
