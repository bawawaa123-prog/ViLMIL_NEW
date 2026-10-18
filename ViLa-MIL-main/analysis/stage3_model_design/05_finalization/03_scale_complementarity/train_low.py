"""Manual Low-only launcher, no overwrite or automatic resume. Default dry-run."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from stage353_common import *


def launch(root, fold):
    parent, path = low_destination(root, fold)
    reserve(parent, private=True)
    lock = parent / '.running.lock'
    lock.touch(exist_ok=False)
    argv = low_command(fold, parent)
    provenance = matched.provenance(argv)
    write_json(parent / 'run_manifest.json', dict(fold=fold, arm='L', split='val', argv=argv,
        inputs=input_hashes(fold), core_code=code_hashes(), protocol=provenance['protocol'],
        package_versions=provenance['package_versions'],
        feature_paths=[str(matched.DATA / n) for n in ('features_biomedclip_5x','features_biomedclip_20x')]))
    print(shlex.join(argv), flush=True)
    with (parent / 'train.log').open('x') as handle:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                handle.write(line); handle.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if code:
        raise RuntimeError(f'Training exited {code}; partial files retained, no auto-resume')
    result = inspect_low(root, fold, require_completion=False)
    if result['status'] != 'Complete':
        raise ValueError(result)
    paths = [parent/'train.log', *path.glob('*.csv'), *path.glob('experiment_*.txt')]
    write_json(parent / 'completion.json', dict(result=result,
        manifest_sha256=matched.digest(parent/'run_manifest.json'),
        checkpoint_identity=routing.checkpoint_identity(result['checkpoint']),
        sha256={str(p.relative_to(parent)):matched.digest(p) for p in paths}))
    lock.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--output-dir', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print('Python:', matched.environment())
    folds = validate_folds(args.folds)
    matched.check_inputs()
    plan = [(f, inspect_low(args.output_dir, f)) for f in folds]
    for f, row in plan:
        if matched.collect('E0', f)['status'] != 'Complete' or row['status']=='Invalid':
            raise ValueError(f'Invalid sources; no training started: {row}')
        print(f'Low fold {f}: {row["status"]}')
        print(shlex.join(low_command(f, low_destination(args.output_dir, f)[0])))
    if not args.execute:
        print('Dry-run. No model construction, training or output creation.'); return
    for f, row in plan:
        if row['status']=='Complete':
            print(f'SKIP verified Low fold {f}')
        else:
            launch(args.output_dir, f)


if __name__ == '__main__':
    main()
