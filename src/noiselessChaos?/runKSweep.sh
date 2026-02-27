#!/bin/bash

# K-sweep at fixed tau=25: check for period doubling near the boundary
# Sweep k from 0.2 to 3 (log-spaced, 60 values)
# eta = 0 (deterministic), one simulation per k value

# Fixed parameters
TAU=25.0
ETA=0.0
THETA0=1.5708
DT=0.01
RECORD_DT=0.1
TMAX=8000.0
SIM_NO=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

# Build log-spaced k array using numpy
read -r -a K_VALUES <<< "$("$PYTHON" -c "
import numpy as np
vals = np.geomspace(0.2, 0.23, 20)
print(' '.join(f'{v:.6f}' for v in vals))
")"

# Get number of processors (keep 2 free)
N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi

# Recompile noiseHeun
echo "Recompiling noiseHeun..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$SCRIPT_DIR/noiseHeun" "$SCRIPT_DIR/noiseHeun.cpp"
echo "Compilation done."

total=${#K_VALUES[@]}
echo "K-sweep at tau=$TAU: ${total} jobs queued"
echo "Running with $N_PROCS parallel jobs..."

export TAU ETA THETA0 DT TMAX SIM_NO RECORD_DT SCRIPT_DIR

if command -v parallel &> /dev/null; then
    for K in "${K_VALUES[@]}"; do
        echo "$K"
    done | parallel -j "$N_PROCS" \
        "$SCRIPT_DIR/noiseHeun" "$TAU" {} "$ETA" "$THETA0" "$DT" "$TMAX" "$SIM_NO" "$RECORD_DT" > /dev/null 2>&1
else
    running=0
    for K in "${K_VALUES[@]}"; do
        "$SCRIPT_DIR/noiseHeun" "$TAU" "$K" \
            "$ETA" "$THETA0" "$DT" "$TMAX" "$SIM_NO" "$RECORD_DT" > /dev/null 2>&1 &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

echo "K-sweep complete."
