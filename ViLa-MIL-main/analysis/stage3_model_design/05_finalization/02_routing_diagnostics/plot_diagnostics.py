"""Bounded statistical figures, no WSI reads. Default dry-run."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from bridge import BASE, matched, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    print('Python:',matched.environment())
    from aggregate_diagnostics import inspect_run
    from run_diagnostics import job_dir
    manifest,state,_ = inspect_run(args.run_dir.resolve())
    required = ['routing_gate_statistics.csv','routing_residual_statistics.csv','routing_intervention_metrics.csv']
    for name in required:
        if not (args.report_dir/name).is_file():
            raise FileNotFoundError(args.report_dir/name)
    report_manifest = __import__('json').loads((args.report_dir/'experiment_manifest.json').read_text())
    if Path(report_manifest['source_run']).resolve()!=args.run_dir.resolve() or report_manifest['source_manifest_sha256']!=matched.digest(args.run_dir/'experiment_manifest.json'):
        raise ValueError('Report and run provenance differ')
    for name in required:
        if report_manifest.get('output_sha256',{}).get(name)!=matched.digest(args.report_dir/name):
            raise ValueError(f'Report table changed since aggregation: {name}')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.execute:
        print('Dry-run: histogram, Gate/fold, intervention differences and one declared spatial sample per fold. No figures generated.')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True,exist_ok=False)
    gate = pd.read_csv(args.report_dir/required[0],dtype={'slide_id':str})
    residual = pd.read_csv(args.report_dir/required[1],dtype={'slide_id':str})
    metrics = pd.read_csv(args.report_dir/required[2])
    hist = []
    for job in manifest['plan']:
        if job['arm']!='E1' or job['mode']!='normal':
            continue
        path = job_dir(args.run_dir,job)/'histograms.csv'
        key = (state.fold==job['fold'])&(state.arm=='E1')&(state['mode']=='normal')
        if state.loc[key,'status'].eq('Complete').all() and path.is_file():
            hist.append(pd.read_csv(path))
    if hist:
        hist = pd.concat(hist,ignore_index=True)
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        for ax,quantity,label in zip(axes,['gate','residual_ratio'],['Mapped gate (not lesion probability)','Per-patch residual / High norm']):
            for fold,part in hist[hist.quantity==quantity].groupby('fold'):
                counts = part.groupby(['bin_left','bin_right'])['count'].sum()
                x = np.array([(a+b)/2 for a,b in counts.index])
                if counts.sum():
                    ax.step(x,counts.to_numpy()/counts.sum(),where='mid',label=f'Fold {fold}')
            ax.set(xlabel=label,ylabel='Patch frequency')
            ax.legend()
        fig.tight_layout(); fig.savefig(args.output_dir/'gate_residual_histograms.png',dpi=180); plt.close(fig)
    if not gate.empty:
        normal = gate[gate['mode']=='normal']
        fig,axes = plt.subplots(1,3,figsize=(12,4))
        for ax,column in zip(axes,['gate_mean','gate_variance','gate_near_one_fraction']):
            folds = sorted(normal.fold.unique())
            values = [normal.loc[normal.fold==f,column].dropna().to_numpy() for f in folds]
            ax.boxplot(values); ax.set_xticks(range(1,len(folds)+1)); ax.set_xticklabels([str(f) for f in folds])
            ax.set(xlabel='Fold',ylabel=column)
        fig.tight_layout(); fig.savefig(args.output_dir/'fold_gate_distributions.png',dpi=180); plt.close(fig)
    if not metrics.empty:
        fig,axes = plt.subplots(1,3,figsize=(12,4))
        for ax,column in zip(axes,['delta_auc','delta_accuracy','delta_macro_f1']):
            for (mode,seed),part in metrics[metrics['mode']!='normal'].groupby(['mode','seed']):
                ax.plot(part.fold,100*part[column],marker='o',label=f'{mode}/{seed}')
            ax.axhline(0,color='black',linewidth=.8); ax.set(xlabel='Fold',ylabel=column+' (pp)')
        axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(args.output_dir/'intervention_deltas.png',dpi=180); plt.close(fig)
    examples = []
    # Deterministic first Slide ID, never pick the strongest effect after inspecting outcomes.
    for job in manifest['plan']:
        if job['arm']!='E1' or job['mode']!='normal':
            continue
        key=(state.fold==job['fold'])&(state.arm=='E1')&(state['mode']=='normal')
        if not state.loc[key,'status'].eq('Complete').all():
            continue
        for filename,column,label in [('patch_samples.csv','gate','High patch'),('region_samples.csv','gate_mean','Low parent')]:
            path=job_dir(args.run_dir,job)/filename
            if not path.is_file():
                continue
            frame=pd.read_csv(path,dtype={'slide_id':str})
            if frame.empty:
                continue
            sid=sorted(frame.slide_id.unique())[0]
            part=frame[frame.slide_id==sid]
            fig,ax=plt.subplots(figsize=(6,5))
            artist=ax.scatter(part.x,part.y,c=part[column],cmap='viridis',vmin=0,vmax=1,s=15)
            ax.invert_yaxis(); ax.set_aspect('equal',adjustable='datalim')
            ax.set(title=f'Fold {job["fold"]}: sampled {label} routing',xlabel='Stored X coordinate',ylabel='Stored Y coordinate')
            fig.colorbar(artist,ax=ax,label=column+' (not lesion probability)')
            fig.tight_layout(); fig.savefig(args.output_dir/f'fold{job["fold"]}_{label.replace(" ","_")}.png',dpi=180); plt.close(fig)
            examples.append(dict(fold=job['fold'],slide_id=sid,source=filename,n_points=len(part)))
    write_json(args.output_dir/'figure_manifest.json',{'run':str(args.run_dir.resolve()),'report':str(args.report_dir.resolve()),
        'status':'Complete figures' if state.status.eq('Complete').all() else 'Partial figures; unfinished jobs excluded',
        'selection':'Lexicographically first Slide ID, fixed bounded samples','spatial_examples':examples,
        'report_hashes':{name:matched.digest(args.report_dir/name) for name in required}})
    print('Figures:',args.output_dir)


if __name__=='__main__':
    main()
