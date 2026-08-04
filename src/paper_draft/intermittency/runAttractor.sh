#!/bin/bash
set -euo pipefail

# Long trajectories either side of the onset, for the attractor panel.
#
# The attractor is drawn in the difference coordinates
#   x = theta(t) - theta(t-tau),   y = theta(t-tau) - theta(t-2*tau),
# which are unchanged by theta -> theta + constant.  Every burst shifts theta by
# ~2*pi, so in these coordinates all the shifted copies of the laminar orbit fall
# on top of each other and the laminar core is a single sharp closed curve --
# no detrending needed, just the symmetry of the equation.
#
# Long runs and a coarse recording step: the density needs many points spread
# over the attractor, not a finely resolved single orbit.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/timeseries"
OUT_DIR="$SCRIPT_DIR/outputs/attractor"

TAU=1.0
DT=0.001
THETA0=1.0
T_WARMUP=20000
T_MEASURE=60000
RECORD_DT=0.02

# Below the onset and just above it, as in the first two timeseries panels.
TAUKS=(4.9700 5.0000)

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
