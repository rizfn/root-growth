#!/bin/bash

# Lyapunov exponent sweep over k at fixed tau=25
# Same k range as the period-doubling k_sweep: geomspace(0.2, 0.23, 20)
# Deterministic (eta=0): warmup 2000 time units, then measure 5000 time units

# Fixed parameters
TAU=25.0
THETA0=1.5708
DT=0.01
RECORD_DT=0.1
T_WARMUP=2000.0
T_LYAP=5000.0
N_PHASES=4   # number of independent attractor phases per k

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

# Build log-spaced k array using numpy (same as runKSweep.sh)
read -r -a K_VALUES <<< "$("$PYTHON" -c "
import numpy as np
vals = np.geomspace(0.06, 0.6, 40)
print(' '.join(f'{v:.6f}' for v in vals))
")"

# Get number of processors (keep 2 free)
N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi

# Recompile lyapunov
echo "Recompiling lyapunov..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$SCRIPT_DIR/lyapunov" "$SCRIPT_DIR/lyapunov.cpp"
echo "Compilation done."

total=${#K_VALUES[@]}
echo "Lyapunov k-sweep at tau=$TAU: ${total} k-values × ${N_PHASES} phases = $((total * N_PHASES)) jobs queued"
echo "Running with $N_PROCS parallel jobs..."

export TAU THETA0 DT T_WARMUP T_LYAP RECORD_DT SCRIPT_DIR

# Build flat list of (K, WARMUP) pairs: warmup offsets are multiples of tau
# spaced by TAU so each phase starts at a different point on the attractor
JOB_LIST=()
for K in "${K_VALUES[@]}"; do
    for ((p=0; p<N_PHASES; p++)); do
        # Use python/bc to compute warmup + p*tau
        WARMUP=$("$PYTHON" -c "print($T_WARMUP + $p * $TAU)")
        JOB_LIST+=("$K $WARMUP")
    done
done

if command -v parallel &> /dev/null; then
    for JOB in "${JOB_LIST[@]}"; do
        echo "$JOB"
    done | parallel -j "$N_PROCS" --colsep ' ' \
        "$SCRIPT_DIR/lyapunov" "$TAU" {1} "$THETA0" "$DT" {2} "$T_LYAP" "$RECORD_DT"
else
    running=0
    for JOB in "${JOB_LIST[@]}"; do
        K=$(echo $JOB | cut -d' ' -f1)
        WARMUP=$(echo $JOB | cut -d' ' -f2)
        "$SCRIPT_DIR/lyapunov" "$TAU" "$K" \
            "$THETA0" "$DT" "$WARMUP" "$T_LYAP" "$RECORD_DT" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

echo "Lyapunov k-sweep complete."
