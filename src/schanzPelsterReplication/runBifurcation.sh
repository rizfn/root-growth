#!/bin/bash
# ============================================================================
# runBifurcation.sh  –  Sweep for Schanz-Pelster bifurcation-diagram data
# ============================================================================
# DDE:  dθ/dt = -k·sin(θ(t-τ))
#
# τ·k landmarks (from Schanz & Pelster, PRE 67, 056205, 2003):
#   π/2 ≈ 1.5708  :  Hopf bifurcation
#   3.77           :  first limit-cycle splits into 2
#   4.105          :  new (second-family) limit cycle appears
#   4.11           :  second family splits
#   4.11-4.175     :  period-doubling cascade
#   4.175-4.24     :  band-merging / chaos
#   4.24           :  end of first chaotic window
#   4.85           :  two new period-2 cycles
#   5.30           :  onset of laminar-phase / phase-slipping chaos
#
# We run three sweeps:
#   1. COARSE  :  τ·k ∈ [1.4, 5.5], ~200 pts  →  full overview
#   2. FINE    :  τ·k ∈ [4.08, 4.30], ~300 pts  →  resolve PD cascade
#   3. POINTS  :  specific τ·k values for time-series and power spectra
#
# Five initial conditions per parameter point:
#   ic = {+0.5, +1.0, -1.0, +2.0, -2.0}
#   The +1.0 / -1.0 family tracks the first limit-cycle branch.
#   The +2.0 / -2.0 family tracks the second (coexisting) branch.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../root-env/bin/python"

TAU=25
DT=0.01
RECORD_DT=0.05   # dense enough for good extremum detection

# Warmup / measure times
T_WARMUP_COARSE=8000
T_MEASURE_COARSE=20000

T_WARMUP_FINE=20000
T_MEASURE_FINE=100000

# ICs for the two attractor families
IC_VALUES=(0.5 1.0 -1.0 2.0 -2.0)

# ── Compile ───────────────────────────────────────────────────────────────────
echo "Compiling bifurcation..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$SCRIPT_DIR/bifurcation" "$SCRIPT_DIR/bifurcation.cpp"
echo "Done."

N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then N_PROCS=1; fi
echo "Using $N_PROCS parallel jobs."

# ── Helper: run one (k, ic) pair ─────────────────────────────────────────────
run_one()
{
    local K="$1" IC="$2" TW="$3" TM="$4"
    "$SCRIPT_DIR/bifurcation" "$TAU" "$K" "$IC" "$DT" "$TW" "$TM" "$RECORD_DT"
}
export -f run_one
export SCRIPT_DIR TAU DT RECORD_DT

# ── 1. COARSE sweep ───────────────────────────────────────────────────────────
read -r -a K_COARSE <<< "$("$PYTHON" -c "
import numpy as np
vals = np.linspace(1.4, 5.5, 210) / $TAU
print(' '.join(f'{v:.8f}' for v in vals))
")"

echo ""
echo "=== COARSE sweep: ${#K_COARSE[@]} k-values × ${#IC_VALUES[@]} ICs ==="
echo "    τ·k ∈ [1.4, 5.5],  t_warmup=$T_WARMUP_COARSE,  t_measure=$T_MEASURE_COARSE"

export T_WARMUP_COARSE T_MEASURE_COARSE
PAIRS_COARSE=()
for K in "${K_COARSE[@]}"; do
    for IC in "${IC_VALUES[@]}"; do
        PAIRS_COARSE+=("$K $IC")
    done
done

if command -v parallel &>/dev/null; then
    printf '%s\n' "${PAIRS_COARSE[@]}" | \
        parallel -j "$N_PROCS" --colsep ' ' \
        run_one {1} {2} "$T_WARMUP_COARSE" "$T_MEASURE_COARSE"
else
    running=0
    for PAIR in "${PAIRS_COARSE[@]}"; do
        read -r K IC <<< "$PAIR"
        run_one "$K" "$IC" "$T_WARMUP_COARSE" "$T_MEASURE_COARSE" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then wait -n; running=$((running - 1)); fi
    done
    wait
fi
echo "COARSE sweep done."

# ── 2. FINE sweep (period-doubling window) ────────────────────────────────────
read -r -a K_FINE <<< "$("$PYTHON" -c "
import numpy as np
vals = np.linspace(4.08, 4.30, 600) / $TAU
print(' '.join(f'{v:.8f}' for v in vals))
")"

echo ""
echo "=== FINE sweep: ${#K_FINE[@]} k-values × ${#IC_VALUES[@]} ICs ==="
echo "    τ·k ∈ [4.08, 4.30],  t_warmup=$T_WARMUP_FINE,  t_measure=$T_MEASURE_FINE"

export T_WARMUP_FINE T_MEASURE_FINE
PAIRS_FINE=()
for K in "${K_FINE[@]}"; do
    for IC in "${IC_VALUES[@]}"; do
        PAIRS_FINE+=("$K $IC")
    done
done

if command -v parallel &>/dev/null; then
    printf '%s\n' "${PAIRS_FINE[@]}" | \
        parallel -j "$N_PROCS" --colsep ' ' \
        run_one {1} {2} "$T_WARMUP_FINE" "$T_MEASURE_FINE"
else
    running=0
    for PAIR in "${PAIRS_FINE[@]}"; do
        read -r K IC <<< "$PAIR"
        run_one "$K" "$IC" "$T_WARMUP_FINE" "$T_MEASURE_FINE" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then wait -n; running=$((running - 1)); fi
    done
    wait
fi
echo "FINE sweep done."

# ── 3. Key points — long timeseries for power spectra & return maps ───────────
# τ·k values: below Hopf, period-1, PD1, PD2, chaos, end-chaos, laminar-onset
read -r -a K_KEY_TAUKVALS <<< "1.4 2.5 3.5 3.77 4.0 4.10 4.108 4.13 4.157 4.165 4.1725 4.178 4.20 4.24 4.30 4.85 5.30"

T_WARMUP_KEY=10000
T_MEASURE_KEY=100000
RECORD_DT_KEY=0.1

echo ""
echo "=== KEY timeseries: ${#K_KEY_TAUKVALS[@]} τ·k values × ${#IC_VALUES[@]} ICs ==="
echo "    t_warmup=$T_WARMUP_KEY,  t_measure=$T_MEASURE_KEY"

PAIRS_KEY=()
for TAUKVAL in "${K_KEY_TAUKVALS[@]}"; do
    K_VAL=$("$PYTHON" -c "print(f'{$TAUKVAL / $TAU:.8f}')")
    for IC in "${IC_VALUES[@]}"; do
        PAIRS_KEY+=("$K_VAL $IC $T_WARMUP_KEY $T_MEASURE_KEY $RECORD_DT_KEY")
    done
done

if command -v parallel &>/dev/null; then
    printf '%s\n' "${PAIRS_KEY[@]}" | \
        parallel -j "$N_PROCS" --colsep ' ' \
        "$SCRIPT_DIR/bifurcation" "$TAU" {1} {2} "$DT" {3} {4} {5}
else
    running=0
    for PAIR in "${PAIRS_KEY[@]}"; do
        read -r K IC TW TM RDDT <<< "$PAIR"
        "$SCRIPT_DIR/bifurcation" "$TAU" "$K" "$IC" "$DT" "$TW" "$TM" "$RDDT" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then wait -n; running=$((running - 1)); fi
    done
    wait
fi
echo "KEY timeseries done."

# ── 4. KEY full timeseries (for power spectra & detailed plots) ───────────────
echo ""
echo "Compiling timeseries..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$SCRIPT_DIR/timeseries" "$SCRIPT_DIR/timeseries.cpp"
echo "Done."

# Same key τ·k values; long t_measure for clean spectra.
T_WARMUP_TS=10000
T_MEASURE_TS=500000
RECORD_DT_TS=0.1

echo ""
echo "=== KEY timeseries: ${#K_KEY_TAUKVALS[@]} τ·k values × ${#IC_VALUES[@]} ICs ==="
echo "    t_warmup=$T_WARMUP_TS,  t_measure=$T_MEASURE_TS,  record_dt=$RECORD_DT_TS"

PAIRS_TS=()
for TAUKVAL in "${K_KEY_TAUKVALS[@]}"; do
    K_VAL=$("$PYTHON" -c "print(f'{$TAUKVAL / $TAU:.8f}')")
    for IC in "${IC_VALUES[@]}"; do
        PAIRS_TS+=("$K_VAL $IC $T_WARMUP_TS $T_MEASURE_TS $RECORD_DT_TS")
    done
done

if command -v parallel &>/dev/null; then
    printf '%s\n' "${PAIRS_TS[@]}" | \
        parallel -j "$N_PROCS" --colsep ' ' \
        "$SCRIPT_DIR/timeseries" "$TAU" {1} {2} "$DT" {3} {4} {5}
else
    running=0
    for PAIR in "${PAIRS_TS[@]}"; do
        read -r K IC TW TM RDDT <<< "$PAIR"
        "$SCRIPT_DIR/timeseries" "$TAU" "$K" "$IC" "$DT" "$TW" "$TM" "$RDDT" &
        running=$((running + 1))
        if [ $running -ge $N_PROCS ]; then wait -n; running=$((running - 1)); fi
    done
    wait
fi
echo "KEY timeseries done."

echo ""
echo "All sweeps complete. Run viz_schanz_pelster.py to generate figures."
