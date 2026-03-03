#!/bin/bash

# Intermittency near-k_c sweep
# k_c is approximately 0.179-0.201 (from Lyapunov analysis).
# Fine sweep: linspace(0.179, 0.210, 20) near k_c
# Long t_measure=500000 to accumulate many laminar intervals.

TAU=25
THETA0=1.5708
DT=0.01
RECORD_DT=0.1
T_WARMUP=5000
T_MEASURE=1000000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

read -r -a K_VALUES <<< "$("$PYTHON" -c "
import numpy as np
vals = np.linspace(0.19, 0.21, 30)
print(' '.join(f'{v:.6f}' for v in vals))
")"

N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi

echo "Recompiling intermittency..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$SCRIPT_DIR/intermittency" "$SCRIPT_DIR/intermittency.cpp"
echo "Compilation done."

total=${#K_VALUES[@]}
echo "Intermittency sweep: $total k-values, t_measure=$T_MEASURE each"
echo "Running with $N_PROCS parallel jobs..."

export TAU THETA0 DT T_WARMUP T_MEASURE RECORD_DT SCRIPT_DIR

if command -v parallel &> /dev/null; then
    printf '%s\n' "${K_VALUES[@]}" | \
        parallel -j "$N_PROCS" \
        "$SCRIPT_DIR/intermittency" "$TAU" {} "$THETA0" "$DT" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT"
else
    running=0
    for K in "${K_VALUES[@]}"; do
        "$SCRIPT_DIR/intermittency" "$TAU" "$K" \
            "$THETA0" "$DT" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

echo "Intermittency sweep complete."
