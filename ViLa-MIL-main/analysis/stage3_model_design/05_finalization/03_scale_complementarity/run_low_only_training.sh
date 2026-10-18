#!/usr/bin/env bash
set -euo pipefail
STAGE353_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$STAGE353_DIR/../01_matched_fivefold/activate_env.sh"
cd "$STAGE353_DIR/../../../.."
exec python "$STAGE353_DIR/train_low.py" "$@"
