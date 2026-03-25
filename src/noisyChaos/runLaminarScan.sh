#!/bin/bash

set -euo pipefail

# Laminar-only 2D scan for stochastic SDDE.
# Strategy: one replicate per parameter cell, very long post-warmup measurement.

TAU=1.0
DT=0.01
THETA0=1.5708
T_WARMUP=1000.0
T_LAMINAR=20000.0
RECORD_DT=0.1
LAMINAR_THRESHOLD=1.5707963267948966
MIN_PERIODS=1.0
SAVE_TIMESERIES=0

K_MIN=4.5
K_MAX=5.5
N_K=41
ETA_MIN=1e-6
ETA_MAX=1e0
N_ETA=31

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"
EXEC="$SCRIPT_DIR/sdde_laminar"

print_progress() {
    local completed=$1
    local total=$2
    local pct=0
    if [ "$total" -gt 0 ]; then
        pct=$((100 * completed / total))
    fi
    printf "\rProgress: %3d%% (%d/%d jobs complete)" "$pct" "$completed" "$total"
}

N_PROCS=$(($(nproc) - 2))
if [ "$N_PROCS" -lt 1 ]; then
    N_PROCS=1
fi

echo "Recompiling sdde_laminar..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$EXEC" "$SCRIPT_DIR/sdde_laminar.cpp"
echo "Compilation done."

read -r -a K_VALUES <<< "$($PYTHON -c "
import numpy as np
vals = np.linspace($K_MIN, $K_MAX, $N_K)
print(' '.join(f'{v:.8f}' for v in vals))
")"

read -r -a ETA_VALUES <<< "$($PYTHON -c "
import numpy as np
vals = np.geomspace($ETA_MIN, $ETA_MAX, $N_ETA)
print(' '.join(f'{v:.10e}' for v in vals))
")"

MANIFEST_DIR="$SCRIPT_DIR/outputs/laminar_scan"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/scan_manifest.tsv"

echo -e "tau\tdt\ttheta0\tt_warmup\tt_laminar\trecord_dt\tlaminar_threshold\tmin_periods\tn_k\tn_eta\tn_jobs" > "$MANIFEST"
TOTAL_JOBS=$((N_K * N_ETA))
echo -e "$TAU\t$DT\t$THETA0\t$T_WARMUP\t$T_LAMINAR\t$RECORD_DT\t$LAMINAR_THRESHOLD\t$MIN_PERIODS\t$N_K\t$N_ETA\t$TOTAL_JOBS" >> "$MANIFEST"

echo "Laminar scan: ${N_K} k-values x ${N_ETA} eta-values = ${TOTAL_JOBS} jobs"
echo "Running with $N_PROCS parallel jobs"

JOB_LIST=()
for ((ik=0; ik<${#K_VALUES[@]}; ik++)); do
    for ((ie=0; ie<${#ETA_VALUES[@]}; ie++)); do
        seed=$((3000001 + ik * 30011 + ie * 353))
        JOB_LIST+=("${K_VALUES[$ik]} ${ETA_VALUES[$ie]} $seed")
    done
done

if command -v parallel >/dev/null 2>&1; then
    JOBLOG=$(mktemp)
    trap 'rm -f "$JOBLOG"' EXIT

    printf '%s\n' "${JOB_LIST[@]}" | parallel -j "$N_PROCS" --colsep ' ' --joblog "$JOBLOG" \
        "$EXEC" "$TAU" {1} {2} "$THETA0" "$DT" "$T_WARMUP" "$T_LAMINAR" "$RECORD_DT" "$LAMINAR_THRESHOLD" "$MIN_PERIODS" {3} "$SAVE_TIMESERIES" &
    PARALLEL_PID=$!

    while kill -0 "$PARALLEL_PID" 2>/dev/null; do
        completed=$(($(wc -l < "$JOBLOG") - 1))
        if [ "$completed" -lt 0 ]; then
            completed=0
        fi
        print_progress "$completed" "$TOTAL_JOBS"
        sleep 1
    done

    wait "$PARALLEL_PID"
    print_progress "$TOTAL_JOBS" "$TOTAL_JOBS"
    printf "\n"
else
    running=0
    completed=0
    for JOB in "${JOB_LIST[@]}"; do
        K=$(echo "$JOB" | cut -d' ' -f1)
        ETA=$(echo "$JOB" | cut -d' ' -f2)
        SEED=$(echo "$JOB" | cut -d' ' -f3)

        "$EXEC" "$TAU" "$K" "$ETA" "$THETA0" "$DT" "$T_WARMUP" "$T_LAMINAR" "$RECORD_DT" "$LAMINAR_THRESHOLD" "$MIN_PERIODS" "$SEED" "$SAVE_TIMESERIES" &

        running=$((running + 1))
        if [ "$running" -ge "$N_PROCS" ]; then
            wait -n
            running=$((running - 1))
            completed=$((completed + 1))
            print_progress "$completed" "$TOTAL_JOBS"
        fi
    done

    while [ "$running" -gt 0 ]; do
        wait -n
        running=$((running - 1))
        completed=$((completed + 1))
        print_progress "$completed" "$TOTAL_JOBS"
    done
    printf "\n"
fi

echo "Laminar scan complete."
