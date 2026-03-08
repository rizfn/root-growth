#!/bin/bash
# ============================================================================
# runTimeseriesLC2.sh  –  Dense τ·k sweep targeting the second limit cycle
# ============================================================================
# DDE:  dθ/dt = -k·sin(θ(t-τ))
#
# Sweeps τ·k ∈ [4.1, 4.11] with 100 points, IC = {+2, -2}.
# IC = ±2 (> π ≈ 3.14) is needed to seed the second-LC basin of attraction.
# Output goes to outputs/timeseries/ (IC is encoded in the filename, so it
# does not collide with the IC=±1 runs from runTimeseries.sh).
# Consumed by viz_lc2_period_amplitude.py.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

TAU=25
DT=0.01
RECORD_DT=0.1

T_WARMUP=10000
T_MEASURE=50000

IC_VALUES=(2.0 -2.0)

# ── Compile ───────────────────────────────────────────────────────────────────
echo "Compiling timeseries..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$SCRIPT_DIR/timeseries" "$SCRIPT_DIR/timeseries.cpp"
echo "Done."

N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi
echo "Using $N_PROCS parallel jobs."

# ── Build k-value list ────────────────────────────────────────────────────────
read -r -a K_VALS <<< "$("$PYTHON" -c "
import numpy as np
vals = np.linspace(4.1, 4.11, 100) / $TAU
print(' '.join(f'{v:.8f}' for v in vals))
")"

echo ""
echo "=== LC2 timeseries sweep: ${#K_VALS[@]} k-values × ${#IC_VALUES[@]} ICs ==="
echo "    τ·k ∈ [4.1, 4.11],  t_warmup=$T_WARMUP,  t_measure=$T_MEASURE,  record_dt=$RECORD_DT"

PAIRS=()
for K in "${K_VALS[@]}"; do
    for IC in "${IC_VALUES[@]}"; do
        PAIRS+=("$K $IC")
    done
done

run_one()
{
    local K="$1" IC="$2"
    "$SCRIPT_DIR/timeseries" "$TAU" "$K" "$IC" "$DT" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT"
}
export -f run_one
export SCRIPT_DIR TAU DT T_WARMUP T_MEASURE RECORD_DT

if command -v parallel &>/dev/null; then
    printf '%s\n' "${PAIRS[@]}" | \
        parallel -j "$N_PROCS" --colsep ' ' run_one {1} {2}
else
    running=0
    for PAIR in "${PAIRS[@]}"; do
        read -r K IC <<< "$PAIR"
        run_one "$K" "$IC" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then wait -n; running=$((running - 1)); fi
    done
    wait
fi

echo ""
echo "Sweep complete. Run viz_lc2_period_amplitude.py to generate plots."
