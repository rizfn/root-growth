#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/bifurcation_target_ic"
OUT_DIR="$SCRIPT_DIR/outputs/bifurcation4IC_target"

TAU=25
DT=0.01
RECORD_DT=0.05

# Target-k warmup then real-k warmup then measurement.
T_WARMUP_TARGET=3000
T_WARMUP_REAL=12000
T_MEASURE=40000

TAUK_MIN=0.001
TAUK_MAX=5.0
N_POINTS=400

# LC2 guidance from your two references:
# - Appearance side uses seed around tau*k~4.13
# - Disappearance side (chaotic LC2) uses seed around tau*k~4.24
LC2_TARGET_APPEAR=4.13
LC2_TARGET_DISAPPEAR=4.24
LC2_APPEAR_CUTOFF=3.985
LC2_DISAPPEAR_CUTOFF=4.24

mkdir -p "$OUT_DIR"

echo "Compiling bifurcation_target_ic.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$EXE" "$SCRIPT_DIR/bifurcation_target_ic.cpp"
echo "Done."

N_PROCS=$(($(nproc) - 2))
if [ "$N_PROCS" -lt 1 ]; then N_PROCS=1; fi
echo "Using $N_PROCS parallel jobs (keeping 2 CPUs free)."

read -r -a K_VALS <<< "$(python3 - <<PY
import numpy as np
vals = np.linspace($TAUK_MIN, $TAUK_MAX, $N_POINTS) / $TAU
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

PAIRS=()
for K in "${K_VALS[@]}"; do
    TAUK=$(awk -v k="$K" -v tau="$TAU" 'BEGIN{printf "%.6f", k*tau}')

    # Branch 1: period-4 from +1
    TGT1="$TAUK"
    PAIRS+=("$K $TGT1 1.0 p4_pos")

    # Branch 2: period-4 from -1
    TGT2="$TAUK"
    PAIRS+=("$K $TGT2 -1.0 p4_neg")

    # Branch 3: LC2 from +2 with guided target selection
    TGT3=$(python3 - <<PY
x=$TAUK
if x < $LC2_APPEAR_CUTOFF:
    print(f"{$LC2_TARGET_APPEAR:.6f}")
elif x > $LC2_DISAPPEAR_CUTOFF:
    print(f"{$LC2_TARGET_DISAPPEAR:.6f}")
else:
    print(f"{x:.6f}")
PY
)
    PAIRS+=("$K $TGT3 2.0 lc2_pos")

    # Branch 4: LC2 from -2 with same guided target selection
    TGT4="$TGT3"
    PAIRS+=("$K $TGT4 -2.0 lc2_neg")
done

run_one() {
    local K_REAL="$1"
    local TAUK_TARGET="$2"
    local THETA0="$3"
    local BRANCH="$4"

    local K_TARGET
    K_TARGET=$(awk -v tkt="$TAUK_TARGET" -v tau="$TAU" 'BEGIN{printf "%.10f", tkt/tau}')

    local TAUK_REAL
    TAUK_REAL=$(awk -v k="$K_REAL" -v tau="$TAU" 'BEGIN{printf "%.6f", k*tau}')

    local TAG_REAL TAG_TARGET TAG_IC OUT_FILE
    TAG_REAL="$(printf '%.6f' "$TAUK_REAL" | tr '-' 'n')"
    TAG_TARGET="$(printf '%.6f' "$TAUK_TARGET" | tr '-' 'n')"
    TAG_IC="$(printf '%.1f' "$THETA0" | tr '-' 'n')"

    OUT_FILE="$OUT_DIR/${BRANCH}_tauk_${TAG_REAL}_target_${TAG_TARGET}_ic_${TAG_IC}.tsv"

    "$EXE" \
        "$TAU" "$K_REAL" "$K_TARGET" "$THETA0" "$DT" \
        "$T_WARMUP_TARGET" "$T_WARMUP_REAL" "$T_MEASURE" "$RECORD_DT" "$OUT_FILE"
}

export -f run_one
export EXE OUT_DIR TAU DT T_WARMUP_TARGET T_WARMUP_REAL T_MEASURE RECORD_DT

echo ""
echo "Running ${#PAIRS[@]} jobs (${#K_VALS[@]} k-values x 4 branches) ..."

if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${PAIRS[@]}" | parallel -j "$N_PROCS" --colsep ' ' run_one {1} {2} {3} {4}
else
    running=0
    for PAIR in "${PAIRS[@]}"; do
        read -r K_REAL TAUK_TARGET THETA0 BRANCH <<< "$PAIR"
        run_one "$K_REAL" "$TAUK_TARGET" "$THETA0" "$BRANCH" &
        running=$((running + 1))
        if [ "$running" -ge "$N_PROCS" ]; then
            wait -n
            running=$((running - 1))
        fi
    done
    wait
fi

echo ""
echo "Done. Data saved in: $OUT_DIR"
echo "Next: python3 $SCRIPT_DIR/viz_bifurcation_4ic_target.py"
