#!/usr/bin/env bash
# Sourced by wrappers. Never substitutes another Python environment.
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x /opt/conda/bin/conda ]]; then
    export PATH="/opt/conda/bin:$PATH"
  else
    echo 'Conda not found; initialize Conda before running.' >&2
    return 1
  fi
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vila_mil_overlay_rt
python -c 'import sys; from pathlib import Path; print(sys.executable); assert Path(sys.prefix).name == "vila_mil_overlay_rt"'
