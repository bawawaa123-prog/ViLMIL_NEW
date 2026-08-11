#!/usr/bin/env bash
set -euo pipefail

# Phase A helper for a single CellSens VSI sample.
# 1) Run showinf and dump series metadata
# 2) Pick the largest series as candidate WSI series
# 3) Convert a small crop (sanity check)
# 4) Optionally convert full series
#
# Usage:
#   bash tools/vsi_phaseA_single.sh \
#     --slide-id 20251222-06 \
#     --data-root /private/ljh-data/shared/data/jiangxi \
#     --work-dir /private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/phaseA_jiangxi \
#     --env vsi_extval
#
# Optional full conversion:
#   ... --full-convert

SLIDE_ID=""
DATA_ROOT="/private/ljh-data/shared/data/jiangxi"
WORK_DIR="/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/data/phaseA_jiangxi"
ENV_NAME="vsi_extval"
CROP_W=4096
CROP_H=4096
FULL_CONVERT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slide-id)
      SLIDE_ID="$2"; shift 2;;
    --data-root)
      DATA_ROOT="$2"; shift 2;;
    --work-dir)
      WORK_DIR="$2"; shift 2;;
    --env)
      ENV_NAME="$2"; shift 2;;
    --crop-w)
      CROP_W="$2"; shift 2;;
    --crop-h)
      CROP_H="$2"; shift 2;;
    --full-convert)
      FULL_CONVERT=1; shift 1;;
    *)
      echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -z "$SLIDE_ID" ]]; then
  echo "--slide-id is required" >&2
  exit 2
fi

BFDIR="/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/tools/bftools/bftools"
SHOWINF="$BFDIR/showinf"
BFCONVERT="$BFDIR/bfconvert"

if [[ ! -x "$SHOWINF" || ! -x "$BFCONVERT" ]]; then
  echo "bftools not found: $BFDIR" >&2
  exit 1
fi

VSI_PATH="$DATA_ROOT/${SLIDE_ID}.vsi"
COMPANION_DIR="$DATA_ROOT/_${SLIDE_ID}_"
if [[ ! -f "$VSI_PATH" ]]; then
  echo "Missing VSI: $VSI_PATH" >&2
  exit 1
fi
if [[ ! -d "$COMPANION_DIR" ]]; then
  echo "Missing companion dir: $COMPANION_DIR" >&2
  exit 1
fi

OUT_DIR="$WORK_DIR/$SLIDE_ID"
mkdir -p "$OUT_DIR"

SHOWINF_TXT="$OUT_DIR/${SLIDE_ID}_showinf.txt"
SERIES_TSV="$OUT_DIR/${SLIDE_ID}_series.tsv"
SERIES_SORTED_TSV="$OUT_DIR/${SLIDE_ID}_series_sorted.tsv"
CROP_OUT="$OUT_DIR/${SLIDE_ID}_series_candidate_crop_${CROP_W}x${CROP_H}.ome.tif"
FULL_OUT="$OUT_DIR/${SLIDE_ID}_series_candidate_full.ome.tif"

echo "[1/5] showinf metadata dump"
conda run -n "$ENV_NAME" "$SHOWINF" -no-upgrade -nopix "$VSI_PATH" > "$SHOWINF_TXT" 2>&1

echo "[2/5] parse series list"
PARSE_PY="$OUT_DIR/parse_series.py"
cat > "$PARSE_PY" <<'PY'
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
args = ap.parse_args()

sid = None
w = None
h = None
rows = []
series_re = re.compile(r"^Series\s+#(\d+)\s*:")

with open(args.src, "r", encoding="utf-8", errors="ignore") as f:
  for line in f:
    m = series_re.match(line.strip())
    if m:
      sid = int(m.group(1))
      w = None
      h = None
      continue
    s = line.strip()
    if s.startswith("Width ="):
      try:
        w = int(s.split("=")[1].strip())
      except Exception:
        w = None
    elif s.startswith("Height ="):
      try:
        h = int(s.split("=")[1].strip())
      except Exception:
        h = None
    elif s.startswith("Pixel type ="):
      if sid is not None and w is not None and h is not None:
        rows.append((sid, w, h, w * h))

rows.sort(key=lambda x: x[3], reverse=True)

with open(args.dst, "w", encoding="utf-8") as out:
  for sid, w, h, area in rows:
    out.write(f"{sid}\t{w}\t{h}\tarea\t{area}\n")
PY

conda run -n "$ENV_NAME" python "$PARSE_PY" --src "$SHOWINF_TXT" --dst "$SERIES_TSV"

if [[ ! -s "$SERIES_TSV" ]]; then
  echo "Failed to parse series metadata. Inspect: $SHOWINF_TXT" >&2
  exit 1
fi

sort -t $'\t' -k5,5nr "$SERIES_TSV" > "$SERIES_SORTED_TSV"
CANDIDATE_SERIES=$(head -n 1 "$SERIES_SORTED_TSV" | cut -f1)
CANDIDATE_W=$(head -n 1 "$SERIES_SORTED_TSV" | cut -f2)
CANDIDATE_H=$(head -n 1 "$SERIES_SORTED_TSV" | cut -f3)

echo "Candidate series: #$CANDIDATE_SERIES (${CANDIDATE_W}x${CANDIDATE_H})"
echo "Top series by area:"
head -n 8 "$SERIES_SORTED_TSV"

echo "[3/5] convert candidate series small crop for sanity"
conda run -n "$ENV_NAME" "$BFCONVERT" -no-upgrade -overwrite -series "$CANDIDATE_SERIES" -crop "0,0,${CROP_W},${CROP_H}" -bigtiff -compression LZW "$VSI_PATH" "$CROP_OUT"

echo "[4/5] verify crop output is readable by OpenSlide"
VERIFY_PY="$OUT_DIR/verify_openslide.py"
cat > "$VERIFY_PY" <<'PY'
import argparse
import openslide
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
args = ap.parse_args()

p = Path(args.path)
slide = openslide.OpenSlide(str(p))
print("openslide_ok", p)
print("dimensions", slide.dimensions)
print("level_count", slide.level_count)
slide.close()
PY

conda run -n "$ENV_NAME" python "$VERIFY_PY" --path "$CROP_OUT"

if [[ "$FULL_CONVERT" -eq 1 ]]; then
  echo "[5/5] full conversion of candidate series (this may be long)"
  conda run -n "$ENV_NAME" "$BFCONVERT" -no-upgrade -overwrite -series "$CANDIDATE_SERIES" -bigtiff -compression LZW "$VSI_PATH" "$FULL_OUT"
  echo "Full output: $FULL_OUT"
else
  echo "[5/5] skip full conversion (use --full-convert to enable)"
fi

echo "Done. Outputs under: $OUT_DIR"
