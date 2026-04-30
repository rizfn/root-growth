#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/ghost_escape_batch"
OUT_DIR="$SCRIPT_DIR/outputs/disappearance"

TAU=1.0
DT=0.001

# Chaotic LC2 disappearance side (ghost above crisis/saddle-node-like threshold)
K_IC=4.24
K_C=4.240415

T_WARMUP=60000
T_MAX_TAU=240000
WIN_TAU=6.0

N_SAMPLES=120
SPACING_TAU=0.1

N_DELTA=40
DELTA_MIN=0.00001
DELTA_MAX=0.01

mkdir -p "$OUT_DIR"

echo "Compiling ghost_escape_batch.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$EXE" "$SCRIPT_DIR/ghost_escape_batch.cpp"
echo "Done."

SAMPLES_OUT="$OUT_DIR/disappearance_samples.tsv"
SUMMARY_OUT="$OUT_DIR/disappearance_summary.tsv"

"$EXE" \
  disappearance \
  "$TAU" "$DT" "$K_IC" "$K_C" "$T_WARMUP" "$T_MAX_TAU" "$WIN_TAU" \
  "$N_SAMPLES" "$SPACING_TAU" "$N_DELTA" "$DELTA_MIN" "$DELTA_MAX" \
  "$SAMPLES_OUT" "$SUMMARY_OUT"

echo "Done."
echo "Outputs:"
echo "  $SAMPLES_OUT"
echo "  $SUMMARY_OUT"
echo "Next: python3 $SCRIPT_DIR/viz_lc2_ghost_escape_times.py"