"""Read validation artifacts only; pending folds never become historical substitutes."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from common import (ARMS, BASE, METRICS, check_inputs, checkpoint_schema,
                    collect, environment, provenance, write_json)


def statistics(metrics):
    statistics_rows = []
    for arm in ARMS:
        rows = metrics[(metrics.arm == arm) & (metrics.status == 'Complete')]
        for metric in METRICS:
            values = rows.set_index('fold')[metric].dropna()
            complete = len(values) == 5
            statistics_rows.append(dict(arm=arm, metric=metric, n=len(values),
                status='Fivefold' if complete else 'Pending (no fivefold statistic)',
                mean=float(values.mean()) if complete else np.nan,
                std=float(values.std(ddof=0)) if complete else np.nan,
                best_fold=int(values.idxmax()) if complete else np.nan,
                worst_fold=int(values.idxmin()) if complete else np.nan))
    return pd.DataFrame(statistics_rows)


def paired(metrics):
    rows, summaries = [], []
    for a, b in [('E1', 'E0'), ('E2', 'E0'), ('E1', 'E2')]:
        diffs = []
        for fold in range(5):
            left = metrics[(metrics.arm == a) & (metrics.fold == fold)].iloc[0]
            right = metrics[(metrics.arm == b) & (metrics.fold == fold)].iloc[0]
            complete = left.status == right.status == 'Complete'
            row = dict(comparison=f'{a}-{b}', fold=fold, status='Complete' if complete else 'Pending')
            for metric in METRICS:
                row[f'delta_{metric}'] = float(left[metric] - right[metric]) if complete else np.nan
            rows.append(row)
            diffs.append(row)
        for metric in METRICS:
            values = pd.Series([r[f'delta_{metric}'] for r in diffs], index=range(5)).dropna()
            complete = len(values) == 5
            summaries.append(dict(comparison=f'{a}-{b}', metric=metric, n_pairs=len(values),
                mean_delta=float(values.mean()) if complete else np.nan,
                improvement_folds=int((values > 0).sum()) if complete else np.nan,
                max_degradation=max(0.0, -float(values.min())) if complete else np.nan,
                worst_delta=float(values.min()) if complete else np.nan,
                worst_fold=int(values.idxmin()) if complete else np.nan,
                status='Fivefold' if complete else 'Pending'))
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def markdown_table(frame):
    # No optional tabulate dependency.
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if pd.isna(value):
                cell = 'Pending'
            elif isinstance(value, (float, np.floating)):
                cell = f'{value:.6f}'
            else:
                cell = str(value).replace('|', '\\|').replace('\n', ' ')
            cells.append(cell)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--validation-dir', type=Path, help='Optional output of evaluate_validation.py')
    args = parser.parse_args()
    print('Python:', environment())
    audit = check_inputs()
    metrics = pd.DataFrame([collect(arm, fold) for arm in ARMS for fold in range(5)])
    metrics = metrics.reindex(columns=list(dict.fromkeys([
        *metrics.columns, 'selected_epoch', 'epochs_run', 'checkpoint', 'metric_source'])))
    if args.validation_dir:
        from common import CSV, LABELS, SPLITS, digest, split_members
        from evaluate_validation import metrics_from_predictions, validation_identity
        labels = pd.read_csv(CSV, dtype=str).set_index('slide_id').label.map(LABELS)
        for i, row in metrics.iterrows():
            if row.status not in ('Complete', 'MissingMetric'):
                continue
            directory = args.validation_dir / row.arm / f'fold{row.fold}'
            if not directory.exists():
                continue
            try:
                replay = json.loads((directory / 'validation_metrics.json').read_text())
                identity = validation_identity(row.arm, int(row.fold), row.to_dict())
                if any(replay.get(k) != v for k, v in identity.items()) or replay.get('strict_load') is not True:
                    raise ValueError('Validation replay provenance/strict loading mismatch')
                predictions_path = directory / 'validation_predictions.csv'
                if digest(predictions_path) != replay['predictions_sha256']:
                    raise ValueError('Validation prediction file changed')
                predictions = pd.read_csv(predictions_path, dtype={'slide_id': str})
                expected = set(split_members(SPLITS / f'splits_{row.fold}.csv')['val'])
                if predictions.slide_id.duplicated().any() or set(predictions.slide_id) != expected:
                    raise ValueError('Replay contains unexpected/missing validation samples')
                if predictions.label.tolist() != labels.loc[predictions.slide_id].tolist():
                    raise ValueError('Replay label mapping differs from dataset CSV')
                actual = metrics_from_predictions(predictions)
                for key in METRICS:
                    if not np.isclose(actual[key], replay[key], atol=1e-12, rtol=0):
                        raise ValueError(f'Replay JSON/predictions disagree: {key}')
                    if np.isfinite(row[key]) and not np.isclose(actual[key], row[key], atol=1e-8, rtol=0):
                        raise ValueError(f'Replay/recorded checkpoint metrics disagree: {key}')
                    metrics.loc[i, key] = actual[key]
                metrics.loc[i, 'status'] = 'Complete'
                metrics.loc[i, 'metric_source'] = str(directory)
            except (ValueError, OSError, KeyError) as exc:
                metrics.loc[i, 'status'] = 'Invalid/Partial'
                metrics.loc[i, 'issue'] = str(exc)
                metrics.loc[i, list(METRICS)] = np.nan
    # Compare shared checkpoint schemas; never ignore unexpected/missing architecture keys.
    schemas = {}
    for i, row in metrics.iterrows():
        if row.status not in ('Complete', 'MissingMetric'):
            continue
        shape = checkpoint_schema(Path(row.checkpoint), row.arm)
        shared = {k: v for k, v in shape.items() if not k.startswith('high_router.')}
        if schemas and shared != next(iter(schemas.values())):
            metrics.loc[i, 'status'] = 'Invalid/Partial'
            metrics.loc[i, 'issue'] = 'Shared checkpoint schema differs across arms/folds'
            metrics.loc[i, list(METRICS)] = np.nan
        else:
            schemas[(row.arm, row.fold)] = shared
    summary = statistics(metrics)
    differences, paired_summary = paired(metrics)
    output = args.output_dir or BASE / 'reports' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(output / 'fivefold_metrics.csv', index=False)
    summary.to_csv(output / 'fivefold_statistics.csv', index=False)
    differences.to_csv(output / 'paired_differences.csv', index=False)
    paired_summary.to_csv(output / 'paired_summary.csv', index=False)
    pd.DataFrame(audit).to_csv(output / 'split_audit.csv', index=False)
    write_json(output / 'audit_manifest.json', provenance([]))
    complete = metrics.status.eq('Complete').all()
    lines = ['# Stage 3.5.1 Matched Five-fold Baseline Validation', '',
        'Validation only. Fold indices below are zero-based. No inference or training was performed.', '',
        'Checkpoint selection: first maximum validation ACC, effective patience=10. '
        'Validation Macro-F1 is read from val_f1 at that exact epoch, never from Test. '
        'AUC uses P(class 1=NonAdenocarcinoma). Macro-F1 uses sklearn average=macro, zero_division=0. '
        'Both validation classes are required. Standard deviation uses ddof=0.', '',
        'All three arms complete.' if complete else 'PENDING: the complete three-arm comparison is unavailable.', '',
        markdown_table(metrics[['arm', 'fold', 'status', *METRICS, 'selected_epoch', 'epochs_run']]), '',
        '## Fivefold statistics', '', markdown_table(summary), '',
        '## Paired fold differences', '', markdown_table(differences), '',
        '## Paired fivefold statistics', '', markdown_table(paired_summary), '',
        '## Provenance and limitations', '',
        'Source paths/checkpoints/issues: fivefold_metrics.csv. Inputs/current sources: audit_manifest.json. '
        'Historical runs have no contemporaneous source/input hashes; semantic config, saved split, '
        'checkpoint schema, epoch sequence and final metrics were audited, but historical bitwise provenance cannot be proved.', '',
        'Strict5 remains a provisional case-level split. Candidate naming groups cross splits; '
        'patient-level independence is unconfirmed. Feature directories retain nominal 5x/20x names '
        'although Stage2 measured approximately 2.5x/10x. No split or feature was changed.', '',
        'Stage 3.5.2 can proceed with all 15 validated folds, matched schemas/configurations and E1 diagnostics. '
        'This is a validation comparison after prior development on the same folds; it is not an untouched test estimate. '
        'Do not tune hyperparameters or select checkpoints using historical Test metrics.']
    issues = metrics[~metrics.status.eq('Complete')][['arm', 'fold', 'status', 'source', 'issue']]
    if not issues.empty:
        lines += ['', '## Missing/invalid artifacts', '', markdown_table(issues)]
    (output / 'fivefold_summary.md').write_text('\n'.join(lines) + '\n')
    print(metrics[['arm', 'fold', 'status', *METRICS]].to_string(index=False))
    print('Report:', output)
    if metrics.status.eq('Invalid/Partial').any() or (args.require_complete and not complete):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
