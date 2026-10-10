#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"
cd "$SCRIPT_DIR/../../../.."
exec python "$SCRIPT_DIR/run_experiment.py" --arm E0 "$@"
