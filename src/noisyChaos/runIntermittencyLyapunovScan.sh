#!/bin/bash

set -euo pipefail

# Intermittency route Lyapunov scan with additive noise.
# Uses intermittencyLyapunov executable (normal form x' = k + x^2 - x^3 + eta dW).

K_MIN=${K_MIN:--0.02}
K_MAX=${K_MAX:-0.02}
N_K=${N_K:-41}
ETA_MIN=${ETA_MIN:-1e-6}
ETA_MAX=${ETA_MAX:-1e0}
N_ETA=${N_ETA:-31}
N_REPS=${N_REPS:-32}

DT=${DT:-0.01}
T_WARMUP=${T_WARMUP:-500.0}
T_LYAP=${T_LYAP:-2000.0}
RENORM_DT=${RENORM_DT:-1.0}
DELTA0=${DELTA0:-1e-8}
X0=${X0:-0.0}
SAVE_TRACE=${SAVE_TRACE:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"
EXEC="$SCRIPT_DIR/intermittencyLyapunov"

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

echo "Recompiling intermittencyLyapunov..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$EXEC" "$SCRIPT_DIR/intermittencyLyapunov.cpp"
echo "Compilation done."

read -r -a K_VALUES <<< "$( $PYTHON - <<PY
import numpy as np
vals = np.linspace($K_MIN, $K_MAX, $N_K)
print(' '.join(f'{v:.8f}' for v in vals))
PY
)"

read -r -a ETA_VALUES <<< "$( $PYTHON - <<PY
import numpy as np
vals = np.geomspace($ETA_MIN, $ETA_MAX, $N_ETA)
print(' '.join(f'{v:.10e}' for v in vals))
PY
)"

MANIFEST_DIR="$SCRIPT_DIR/outputs/intermittency_lyapunov"
mkdir -p "$MANIFEST_DIR"
MANIFEST="$MANIFEST_DIR/scan_manifest.tsv"

echo -e "k\teta\tx0\tdt\tt_warmup\tt_lyap\trenorm_dt\tdelta0\tn_k\tn_eta\tn_reps\tn_jobs" > "$MANIFEST"
TOTAL_JOBS=$((N_K * N_ETA * N_REPS))
echo -e "$K_MIN\t$ETA_MIN\t$X0\t$DT\t$T_WARMUP\t$T_LYAP\t$RENORM_DT\t$DELTA0\t$N_K\t$N_ETA\t$N_REPS\t$TOTAL_JOBS" >> "$MANIFEST"

echo "Intermittency Lyapunov scan: ${N_K} k-values x ${N_ETA} eta-values x ${N_REPS} reps = ${TOTAL_JOBS} jobs"
echo "Running with $N_PROCS parallel jobs"

JOB_LIST=()
for ((ik=0; ik<${#K_VALUES[@]}; ik++)); do
    for ((ie=0; ie<${#ETA_VALUES[@]}; ie++)); do
        for ((rep=0; rep<N_REPS; rep++)); do
            seed=$((3000007 + ik * 20011 + ie * 257 + rep))
            JOB_LIST+=("${K_VALUES[$ik]} ${ETA_VALUES[$ie]} $seed")
        done
    done
done

if command -v parallel >/dev/null 2>&1; then
    JOBLOG=$(mktemp)
    trap 'rm -f "$JOBLOG"' EXIT

    printf '%s\n' "${JOB_LIST[@]}" | parallel -j "$N_PROCS" --colsep ' ' --joblog "$JOBLOG" \
        "$EXEC" {1} {2} "$X0" "$DT" "$T_WARMUP" "$T_LYAP" "$RENORM_DT" "$DELTA0" {3} "$SAVE_TRACE" &
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

        "$EXEC" "$K" "$ETA" "$X0" "$DT" "$T_WARMUP" "$T_LYAP" "$RENORM_DT" "$DELTA0" "$SEED" "$SAVE_TRACE" &

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

echo "Intermittency Lyapunov scan complete."
