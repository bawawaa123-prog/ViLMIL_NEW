"""Read-only input/result audit; no model construction or inference."""
import json
from pathlib import Path
from common import ARMS, check_inputs, collect, environment


def main():
    print('Python:', environment())
    print('Split audit:', json.dumps(check_inputs(), ensure_ascii=False))
    rows = [collect(a, f) for a in ARMS for f in range(5)]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    if any(r['status'] == 'Invalid/Partial' for r in rows):
        raise SystemExit(2)
    print('Preflight complete; pending jobs reported, no training or inference.')


if __name__ == '__main__':
    main()
