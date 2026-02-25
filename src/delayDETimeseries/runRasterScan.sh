#!/bin/bash

# Raster-scan: sweep tau × k on a log-log grid (40 × 40)
# All combinations are run (no oscillation-threshold filter).
# eta = 0 (deterministic), one simulation per parameter pair

# Fixed parameters
ETA=0.0
THETA0=1.5708
DT=0.01
RECORD_DT=0.1
TMAX=4000.0
SIM_NO=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

# Build log-spaced parameter arrays using numpy
# tau: 10 .. 105  (50 values, log-spaced)
# k  : 0.025 .. 0.5 (50 values, log-spaced)
read -r -a TAU_VALUES <<< "$("$PYTHON" -c "
import numpy as np
vals = np.geomspace(10, 105, 50)
print(' '.join(f'{v:.4f}' for v in vals))
")"
read -r -a K_VALUES <<< "$("$PYTHON" -c "
import numpy as np
vals = np.geomspace(0.025, 0.5, 50)
print(' '.join(f'{v:.5f}' for v in vals))
")"

# Get number of processors (keep 2 free)
N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi

# Recompile noiseHeun
echo "Recompiling noiseHeun..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$SCRIPT_DIR/noiseHeun" "$SCRIPT_DIR/noiseHeun.cpp"
echo "Compilation done."

# All (tau, k) pairs — no filtering
declare -a JOBS_TAU
declare -a JOBS_K

for TAU in "${TAU_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        JOBS_TAU+=("$TAU")
        JOBS_K+=("$K")
    done
done

total=${#JOBS_TAU[@]}
echo "Raster scan: ${total} jobs queued (no filtering)"
echo "Running with $N_PROCS parallel jobs..."

export ETA THETA0 DT TMAX SIM_NO RECORD_DT SCRIPT_DIR

if command -v parallel &> /dev/null; then
    for i in "${!JOBS_TAU[@]}"; do
        echo "${JOBS_TAU[$i]} ${JOBS_K[$i]}"
    done | parallel -j "$N_PROCS" --col-sep ' ' \
        "$SCRIPT_DIR/noiseHeun" {1} {2} "$ETA" "$THETA0" "$DT" "$TMAX" "$SIM_NO" "$RECORD_DT" > /dev/null 2>&1
else
    running=0
    for i in "${!JOBS_TAU[@]}"; do
        "$SCRIPT_DIR/noiseHeun" "${JOBS_TAU[$i]}" "${JOBS_K[$i]}" \
            "$ETA" "$THETA0" "$DT" "$TMAX" "$SIM_NO" "$RECORD_DT" > /dev/null 2>&1 &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

echo "Raster scan complete."
