#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/ghost_escape_batch"
OUT_DIR="$SCRIPT_DIR/outputs/arrival"

TAU=1.0
DT=0.001

# LC2 appearance side (ghost below onset)
K_IC=4.13
K_C=3.9849

T_WARMUP=60000
T_MAX_TAU=12000
WIN_TAU=6.0

N_SAMPLES=120
SPACING_TAU=1.0

N_DELTA=36
DELTA_MIN=0.0001
DELTA_MAX=0.1

mkdir -p "$OUT_DIR"

echo "Compiling ghost_escape_batch.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$EXE" "$SCRIPT_DIR/ghost_escape_batch.cpp"
echo "Done."

SAMPLES_OUT="$OUT_DIR/arrival_samples.tsv"
SUMMARY_OUT="$OUT_DIR/arrival_summary.tsv"

"$EXE" \
  arrival \
  "$TAU" "$DT" "$K_IC" "$K_C" "$T_WARMUP" "$T_MAX_TAU" "$WIN_TAU" \
  "$N_SAMPLES" "$SPACING_TAU" "$N_DELTA" "$DELTA_MIN" "$DELTA_MAX" \
  "$SAMPLES_OUT" "$SUMMARY_OUT"

echo "Done."
echo "Outputs:"
echo "  $SAMPLES_OUT"
echo "  $SUMMARY_OUT"
echo "Next: python3 $SCRIPT_DIR/viz_lc2_ghost_escape_times.py"