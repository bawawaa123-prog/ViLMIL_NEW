"""Frozen validation inference. Default dry-run; --execute is required."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from bridge import (BASE, ROOT, MODES, SEEDS, audit, jobs, matched, slide_seed,
                    build_model, checkpoint_identity, write_json, GATE_LOW, GATE_HIGH, NEAR_MARGIN)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--shuffle-seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--include-e2', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--patch-samples', type=int, default=256)
    parser.add_argument('--execute', action='store_true')
    return parser.parse_args()


def job_dir(root, job):
    return root / f'fold{job["fold"]}' / job['arm'] / f'{job["mode"]}_seed{job["seed"]}'


def validate_predictions(frame, fold):
    from evaluate_validation import metrics_from_predictions
    expected = set(matched.split_members(matched.SPLITS / f'splits_{fold}.csv')['val'])
    if frame.slide_id.duplicated().any() or set(frame.slide_id) != expected:
        raise ValueError('Unexpected/missing Validation Slide IDs')
    labels = pd.read_csv(matched.CSV,dtype=str).set_index('slide_id').label.map(matched.LABELS)
    if frame.label.tolist() != labels.loc[frame.slide_id].tolist():
        raise ValueError('Validation labels differ from frozen CSV')
    return metrics_from_predictions(frame)


def parameter_drift(e0, e1):
    import torch
    left = torch.load(e0['checkpoint'], map_location='cpu',weights_only=True,mmap=True)
    right = torch.load(e1['checkpoint'], map_location='cpu',weights_only=True,mmap=True)
    prefixes = ('attention_', 'cross_attention_', 'norm.', 'learnable_image_center')
    rows = []
    for key in left:
        if key != 'prompt_learner.ctx' and not key.startswith(prefixes):
            continue
        a,b = left[key].double().reshape(-1),right[key].double().reshape(-1)
        delta = (b-a).norm().item()
        an,bn = a.norm().item(),b.norm().item()
        rows.append(dict(fold=e0['fold'],parameter=key,n_parameters=a.numel(),
                         l2_difference=delta,relative_l2_to_e0=delta/max(an,1e-8),
                         cosine_similarity=float(torch.dot(a,b)/max(an*bn,1e-8)),
                         e0_checkpoint=e0['checkpoint'],e1_checkpoint=e1['checkpoint'],status='Complete'))
    return rows


def run_fold(args, root, fold, sources, plan):
    import torch
    from tools import stage335_routing_diagnostics as s335
    from tools import stage336_context_shuffle_stability as s336
    from tools import stage337_low_context_spatial_discriminability as s337
    from utils.utils import get_simple_loader
    from telemetry import RouterRecorder
    from types import SimpleNamespace
    s336._seed_everything(1)
    config = SimpleNamespace(task='task_adenocarcinoma',csv_path=str(matched.CSV),
        data_root_dir=str(matched.DATA), data_folder_s='features_biomedclip_5x',
        data_folder_l='features_biomedclip_20x', mapping_path=str(matched.MAPPING),
        text_prompt=None, text_prompt_path=str(matched.PROMPT))
    prompts = s335._prompts(config)
    dataset,_ = s335._dataset(config,prompts)
    split = dataset.return_splits(from_id=False,csv_path=str(matched.SPLITS/f'splits_{fold}.csv'))[1]
    loader = get_simple_loader(split,batch_size=1,num_workers=0,mode='transformer')
    device = s335.DEVICE
    for arm in ('E0','E1','E2') if args.include_e2 else ('E0','E1'):
        row = sources[(arm,fold)]
        model = build_model(row,prompts,device)
        recorder = RouterRecorder(model.high_router) if arm == 'E1' else None
        try:
            for job in [j for j in plan if j['fold']==fold and j['arm']==arm]:
                out = job_dir(root,job)
                out.mkdir(parents=True,exist_ok=False)
                info = {**job,'checkpoint':row['checkpoint'],'status':'Complete','split':'val'}
                predictions, gates, residuals, histograms, samples, regions, low_stats = [],[],[],[],[],[],[]
                if arm=='E1':
                    model.routing_diagnostic_mode = job['mode']
                with torch.no_grad():
                    for batch in loader:
                        sid = str(batch[5][0])
                        seed = slide_seed(job['seed'],fold,sid)
                        tensors = [x.to(device) for x in batch[:5]]
                        mapping = batch[6][0]
                        if arm=='E1':
                            model.routing_diagnostic_seed = seed
                            high_coords = batch[3].squeeze(0) if batch[3].ndim==3 else batch[3]
                            recorder.prepare(mapping,high_coords.cpu().numpy(),seed,args.patch_samples)
                        probability,_,_ = model(*tensors,mapping=mapping)
                        p = probability[0].cpu().numpy()
                        predictions.append({**info,'slide_id':sid,'label':int(tensors[4].item()),
                            'prediction':int(p.argmax()),'p_0':float(p[0]),'p_1':float(p[1]),
                            'signed_margin':float(p[1]-p[0]),'absolute_margin':float(abs(p[1]-p[0])),
                            'true_class_margin':float(p[int(tensors[4].item())]-p[1-int(tensors[4].item())]),
                            'slide_shuffle_seed':seed if job['mode']=='context_shuffle' else 0})
                        if recorder is not None:
                            metadata = {**info,'slide_id':sid}
                            gate_keys = [k for k in recorder.row if 'gate' in k or 'parent_group' in k]
                            gates.append({**metadata,**{k:recorder.row[k] for k in gate_keys}})
                            residuals.append({**metadata,**{k:v for k,v in recorder.row.items() if k not in gate_keys}})
                            histograms.extend({**metadata,**item} for item in recorder.histograms)
                            if job['mode']=='normal':
                                samples.extend({**metadata,**item} for item in recorder.patches)
                                group_rng = np.random.default_rng(seed)
                                selected = group_rng.choice(len(recorder.regions),size=min(args.patch_samples,len(recorder.regions)),replace=False)
                                coords = (batch[1].squeeze(0) if batch[1].ndim==3 else batch[1]).cpu().numpy()
                                regions.extend({**metadata,**recorder.regions[i],
                                    'x':float(coords[recorder.regions[i]['low_parent'],0]),
                                    'y':float(coords[recorder.regions[i]['low_parent'],1])} for i in np.sort(selected))
                                low_tensor = tensors[0].squeeze(0) if tensors[0].ndim==3 and tensors[0].shape[0]==1 else tensors[0]
                                low_stats.append({**metadata,**s337.low_spatial_statistics(low_tensor,mapping,seed=seed,pair_samples=256)})
                frame = pd.DataFrame(predictions)
                metrics = validate_predictions(frame,fold)
                if job['mode']=='normal':
                    for k,v in metrics.items():
                        if not np.isclose(v,row[k],atol=1e-7,rtol=0):
                            raise ValueError(f'{arm}/{fold}: Normal replay differs from Stage351 {k}: {v} vs {row[k]}')
                frame.to_csv(out/'predictions.csv',index=False)
                for filename,records in [('gate_statistics.csv',gates),('residual_statistics.csv',residuals),
                                         ('histograms.csv',histograms),('patch_samples.csv',samples),
                                         ('region_samples.csv',regions),('low_context_statistics.csv',low_stats)]:
                    if records:
                        pd.DataFrame(records).to_csv(out/filename,index=False)
                write_json(out/'completion.json',{**info,'n_slides':len(frame),'metrics':metrics,
                    'checkpoint_identity':checkpoint_identity(row['checkpoint']),
                    'sha256':{p.name:matched.digest(p) for p in out.glob('*.csv')}})
                print(f'Completed {arm}/{fold}/{job["mode"]}/{job["seed"]}: {metrics}',flush=True)
        finally:
            if recorder:
                recorder.close()
            del model
            if device.type=='cuda':
                torch.cuda.empty_cache()
    if ('E0',fold) in sources and ('E1',fold) in sources:
        pd.DataFrame(parameter_drift(sources[('E0',fold)],sources[('E1',fold)])).to_csv(root/f'fold{fold}/parameter_differences.csv',index=False)


class Tee:
    """Borrowed streams: their owners manage closure, including logging shutdown."""
    def __init__(self,terminal,handle):
        self.terminal,self.handle = terminal,handle
    def write(self,text):
        written = self.terminal.write(text)
        if not self.handle.closed:
            self.handle.write(text)
            self.handle.flush()
        return written
    def flush(self):
        if not self.terminal.closed:
            self.terminal.flush()
        if not self.handle.closed:
            self.handle.flush()
    def close(self):
        # absl may retain this wrapper until atexit, after the log context has
        # closed. Never close stdout/stderr or the separately owned log here.
        self.flush()


def main():
    args = parse_args()
    if len(args.folds)!=len(set(args.folds)) or len(args.modes)!=len(set(args.modes)) or len(args.shuffle_seeds)!=len(set(args.shuffle_seeds)):
        raise ValueError('Duplicate folds/modes/seeds are not allowed')
    if 'normal' not in args.modes or args.patch_samples<1 or any(s<0 for s in args.shuffle_seeds):
        raise ValueError('Normal required, positive patch-samples and nonnegative fixed seeds required')
    print('Python:',matched.environment())
    rows,split_audit = audit(args.folds,args.include_e2)
    plan = jobs(args.folds,args.modes,args.shuffle_seeds,args.include_e2)
    root = (args.output_dir or BASE/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if root.exists():
        raise FileExistsError(f'Existing output (complete or partial): {root}; use preflight --run-dir to inspect, never overwrite')
    for job in plan:
        print({**job,'checkpoint':next(r['checkpoint'] for r in rows if (r['arm'],r['fold'])==(job['arm'],job['fold'])),
               'output':str(job_dir(root,job))})
    if not args.execute:
        print('Dry-run complete. No model construction, inference or output-directory creation.')
        return
    root.mkdir(parents=True,exist_ok=False)
    (root/'.running.lock').touch(exist_ok=False)
    provenance = matched.provenance(sys.argv)
    provenance['sha256'].update({str(p):matched.digest(p) for p in BASE.rglob('*') if p.suffix in ('.py','.sh','.md') and 'runs' not in p.parts and 'reports' not in p.parts})
    write_json(root/'experiment_manifest.json',{'status':'Planned','plan':plan,'sources':rows,
        'checkpoint_identities':{r['checkpoint']:checkpoint_identity(r['checkpoint']) for r in rows},
        'folds':args.folds,'modes':args.modes,'shuffle_seeds':args.shuffle_seeds,'split':'val',
        'gate_thresholds':[GATE_LOW,GATE_HIGH],'near_margin':NEAR_MARGIN,'patch_samples':args.patch_samples,
        'provenance':provenance,'split_audit':split_audit})
    with (root/'run.log').open('x',encoding='utf-8') as log:
        oldout,olderr = sys.stdout,sys.stderr
        sys.stdout,sys.stderr = Tee(oldout,log),Tee(olderr,log)
        try:
            sources = {(r['arm'],r['fold']):r for r in rows}
            for fold in args.folds:
                run_fold(args,root,fold,sources,plan)
            final_files = [*(job_dir(root,j)/'completion.json' for j in plan),
                           *(root/f'fold{f}/parameter_differences.csv' for f in args.folds)]
            write_json(root/'completion.json',{'status':'Complete','n_jobs':len(plan),
                'manifest_sha256':matched.digest(root/'experiment_manifest.json'),
                'sha256':{str(p.relative_to(root)):matched.digest(p) for p in final_files}})
            (root/'.running.lock').unlink()
            print('Completed inference. Aggregate this directory:',root)
        except BaseException:
            traceback.print_exc()
            print('Partial output preserved; no automatic resume or overwrite.',file=sys.stderr)
            raise
        finally:
            sys.stdout,sys.stderr = oldout,olderr


if __name__=='__main__':
    main()
