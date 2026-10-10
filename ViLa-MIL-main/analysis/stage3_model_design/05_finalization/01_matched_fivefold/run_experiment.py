"""Manual-only launcher: protected destinations, fail-fast checks, no implicit resume."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from common import (ARMS, BASE, ROOT, check_inputs, collect, command,
                    destinations, environment, inspect_result, provenance, write_json)


def plan(arm):
    folds = range(5) if arm == 'E2' else (2, 3, 4)
    # Validate reused development folds before allowing any continuation.
    if arm != 'E2':
        for fold in (0, 1):
            result = collect(arm, fold)
            if result['status'] not in ('Complete', 'MissingMetric'):
                raise RuntimeError(f'Required reused fold: {result}')
    entries = [(fold, collect(arm, fold)) for fold in folds]
    bad = [row for _, row in entries if row['status'] == 'Invalid/Partial']
    if bad:
        raise RuntimeError(f'Partial/invalid experiments; no jobs started: {bad}')
    return entries


def launch(arm, fold):
    path, log = destinations(arm, fold)[-1]
    parent = path.parent
    # Atomic directory creation protects against concurrent launches and stale logs.
    parent.mkdir(parents=True, exist_ok=False)
    lock = parent / '.running.lock'
    lock.touch(exist_ok=False)
    argv = command(arm, fold, parent)
    write_json(parent / 'run_manifest.json', provenance(argv))
    print(shlex.join(argv), flush=True)
    try:
        with log.open('x', encoding='utf-8') as handle:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    handle.write(line)
                    handle.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f'{arm}/{fold} training exited {code}; preserved partial output at {parent}')
        result = inspect_result(path, log, arm, fold)
        write_json(parent / 'completion.json', result)
        lock.unlink()
    except BaseException:
        print(f'Failed/interrupted: {parent}. Output preserved; automatic resume is forbidden.', file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicit manual launch of formal training')
    args = parser.parse_args()
    print('Python:', environment())
    check_inputs()
    entries = plan(args.arm)
    for fold, row in entries:
        print(f'{args.arm} fold={fold}: {row["status"]} {row["source"]}')
        if row['status'] == 'Pending':
            path, _ = destinations(args.arm, fold)[-1]
            if path.parent.exists():
                raise RuntimeError(f'Existing reserved destination: {path.parent}; will not overwrite')
            print(shlex.join(command(args.arm, fold, path.parent)))
    if not args.execute:
        print('Dry run complete. No training was started.')
        return
    for fold, row in entries:
        if row['status'] in ('Complete', 'MissingMetric'):
            print(f'SKIP verified completed {args.arm}/{fold}; reused {row["source"]}')
        else:
            launch(args.arm, fold)


if __name__ == '__main__':
    main()
