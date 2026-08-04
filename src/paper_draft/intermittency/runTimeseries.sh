#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/timeseries"
OUT_DIR="$SCRIPT_DIR/outputs/timeseries"

TAU=1.0
DT=0.001
THETA0=1.0
T_WARMUP=20000
T_MEASURE=4000
RECORD_DT=0.01

# Below the onset (tau*k_c = 4.98818), just above it, and well above it.
TAUKS=(4.9700 5.0000 5.2000)

mkdir -p "$OUT_DIR"

echo "Compiling timeseries.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O3 -o "$EXE" "$SCRIPT_DIR/timeseries.cpp"
echo "Done."

for tauk in "${TAUKS[@]}"; do
    k=$(awk -v x="$tauk" -v tau="$TAU" 'BEGIN{printf "%.10f", x/tau}')
    out_file="$OUT_DIR/tauk_${tauk}.tsv"
    "$EXE" "$TAU" "$k" "$THETA0" "$DT" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT" "$out_file" &
done
wait

echo ""
echo "Done. Data in: $OUT_DIR"
echo "Next: python3 $SCRIPT_DIR/viz_intermittency.py"
