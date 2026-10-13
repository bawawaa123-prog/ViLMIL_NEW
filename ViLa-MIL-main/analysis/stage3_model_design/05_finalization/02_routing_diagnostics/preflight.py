"""Read-only source/path checks and safe inspection of interrupted diagnostics."""
import argparse
from pathlib import Path
import json
from bridge import BASE, MODES, SEEDS, audit, jobs, matched, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds',type=int,nargs='+',choices=range(5),default=list(range(5)))
    parser.add_argument('--include-e2',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--execute',action='store_true',help='Write a new preparation report only; never inference')
    args = parser.parse_args()
    print('Python:',matched.environment())
    if args.run_dir:
        from aggregate_diagnostics import inspect_run
        manifest,state,_ = inspect_run(args.run_dir.resolve())
        print(state.to_string(index=False))
        print('Root lock:',(args.run_dir/'.running.lock').exists(),'completion:',(args.run_dir/'completion.json').exists())
        if not state.status.eq('Complete').all() or not (args.run_dir/'completion.json').exists() or (args.run_dir/'.running.lock').exists():
            raise SystemExit(2)
        return
    rows,split = audit(args.folds,args.include_e2)
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    print('All requested Stage351 sources verified. Stage352 formal diagnostics remain Pending.')
    if args.execute:
        if not args.output_dir:
            parser.error('--execute preparation report requires a NEW --output-dir')
        args.output_dir.mkdir(parents=True,exist_ok=False)
        plan = jobs(args.folds,MODES,SEEDS,args.include_e2)
        write_json(args.output_dir/'experiment_manifest.json',{'status':'Pending','sources':rows,'plan':plan,'split_audit':split,'split':'val'})
        lines = ['# Stage 3.5.2 preparation status','','Checkpoint/config/split metadata checks passed. Formal Validation inference and all mechanism questions remain Pending.','',
                 '| Fold | Model | Mode | Seed | Status |','|---|---|---|---|---|']
        lines += [f'| {j["fold"]} | {j["arm"]} | {j["mode"]} | {j["seed"]} | Pending |' for j in plan]
        lines += ['','Q1–Q6: insufficient current-checkpoint diagnostic evidence; historical Stage335–337 results are background only.']
        (args.output_dir/'stage352_summary.md').write_text('\n'.join(lines)+'\n')
        print('Preparation report:',args.output_dir)


if __name__=='__main__':
    main()
