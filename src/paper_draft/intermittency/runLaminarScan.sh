#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/laminar_lengths"
OUT_DIR="$SCRIPT_DIR/outputs/laminar_scan"

TAU=1.0
DT=0.001

# Independent trajectories per k: different constant initial histories on the
# same (theta0 in (0, pi)) branch.  Averaging over them smooths the periodic
# windows that are dense just above the onset, and supplies the statistics for
# the k where laminar phases are long.
THETA0_MIN=0.5
THETA0_MAX=2.5
N_RUNS=24

T_WARMUP=2000
T_MEASURE=500000

# Half the ~2*pi spacing between neighbouring laminar plateaus, and the drift of
# the oscillation centre over one period below which a burst counts as over.
SLIP_THRESHOLD=3.14159265
QUIET_THRESHOLD=0.5

# Anchor for the grid only.  Nothing bursts in 24 x 5e5 time units up to 4.9871
# and the first (transient) bursts appear at 4.9875, so the grid is laid out from
# here; the onset that the laminar times actually diverge at is fitted in
# viz_intermittency.py and comes out at 4.98818, which leaves the smallest k of
# the grid at eps ~ 1e-4.
TAUK_C=4.9873
EPS_MIN=0.001
EPS_MAX=0.5
N_EPS=22

mkdir -p "$OUT_DIR"

echo "Compiling laminar_lengths.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O3 -o "$EXE" "$SCRIPT_DIR/laminar_lengths.cpp"
echo "Done."

read -r -a TAUKS <<< "$(python3 - <<PY
import numpy as np
# Below the onset, plus the three tau*k that runTimeseries.sh illustrates, so
# that both figures speak about exactly the same parameter values.
below = [4.9855, 4.9865, 4.9870, 4.9875, 4.9700, 5.0000, 5.2000]
above = $TAUK_C + np.geomspace($EPS_MIN, $EPS_MAX, $N_EPS)
print(' '.join(f'{v:.6f}' for v in np.concatenate([below, above])))
PY
)"

run_one() {
    local tauk="$1"
    local k
    k=$(awk -v x="$tauk" -v tau="$TAU" 'BEGIN{printf "%.10f", x/tau}')
    local out_file="$OUT_DIR/tauk_${tauk}.tsv"

    if [ -f "$out_file" ]; then
        echo "  using existing tau*k=$tauk"
        return
    fi

    "$EXE" \
        "$TAU" "$k" "$THETA0_MIN" "$THETA0_MAX" "$N_RUNS" "$DT" \
        "$T_WARMUP" "$T_MEASURE" "$SLIP_THRESHOLD" "$QUIET_THRESHOLD" "$out_file"
    echo "  done tau*k=$tauk"
}

export -f run_one
export EXE OUT_DIR TAU THETA0_MIN THETA0_MAX N_RUNS DT T_WARMUP T_MEASURE
export SLIP_THRESHOLD QUIET_THRESHOLD

N_PROCS=$(($(nproc) - 2))
if [ "$N_PROCS" -lt 1 ]; then N_PROCS=1; fi
echo "Running ${#TAUKS[@]} k-values on $N_PROCS parallel jobs (keeping 2 CPUs free)."

if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${TAUKS[@]}" | parallel -j "$N_PROCS" run_one {}
else
    printf '%s\n' "${TAUKS[@]}" | xargs -P "$N_PROCS" -I{} bash -c 'run_one "$@"' _ {}
fi

echo ""
echo "Done. Data in: $OUT_DIR"
echo "Next: python3 $SCRIPT_DIR/viz_intermittency.py"
