"""Read-only protocol/source audit; --execute only writes a preparation report."""
import argparse
from pathlib import Path
import pandas as pd
from stage353_common import *


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--low-run-dir', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--stage352-run-dir', type=Path, default=DEFAULT_352_RUN)
    p.add_argument('--stage352-report-dir', type=Path, default=DEFAULT_352_REPORT)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--require-complete', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print('Python:', matched.environment())
    folds = validate_folds(args.folds)
    inputs = matched.check_inputs()
    historical = historical_audit()
    high = [matched.collect('E0', f) for f in folds]
    low = [inspect_low(args.low_run_dir, f) for f in folds]
    manifest, frames = load_stage352(args.stage352_run_dir, args.stage352_report_dir)
    for f, h in zip(folds, high):
        if h['status'] != 'Complete':
            raise ValueError(f'Unavailable High: {h}')
        assert_metrics(validate_predictions(frames[(f,'E0','normal',0)], f, 'E0', h['checkpoint']), h)
    for item in historical:
        print(item['name'], item['checkpoint_folds'], item['status'], item['issues'])
    print(pd.DataFrame(high+low)[['arm','fold','status','source','issue']].to_string(index=False))
    if args.execute:
        if args.output_dir is None:
            p.error('--execute requires --output-dir')
        out = reserve(args.output_dir)
        write_json(out / 'source_audit.json', dict(historical=historical, high=high, low=low,
            split_audit=inputs, stage352_manifest_sha256=matched.digest(args.stage352_run_dir/'experiment_manifest.json'),
            status='Complete sources' if all(r['status']=='Complete' for r in low) else 'Pending Low training'))
        pd.DataFrame(high+low).to_csv(out/'experiment_status.csv', index=False)
        with (out/'stage353_summary.md').open('x') as handle:
            handle.write('# Stage 3.5.3 preparation status\n\nPending：匹配Independent Low五折训练与Validation预测尚未生成，G1–G4、Oracle及Routing关联结果均Pending。\n\nHigh五折E0与Stage352来源已核验。历史Low来源审计详见source_audit.json；不能将旧Test指标填充本阶段Validation结果。形态复核Pending。\n')
    if any(r['status']=='Invalid' for r in low) or (args.require_complete and any(r['status']!='Complete' for r in low)):
        raise SystemExit(1)
    print('Source checks passed; Pending is preparation status, not completed Stage353 evidence.')


if __name__ == '__main__':
    main()
