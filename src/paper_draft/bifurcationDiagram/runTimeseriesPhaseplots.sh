#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/timeseries_phase_target_ic"
OUT_DIR="$SCRIPT_DIR/outputs/timeseries_phaseplots"

TAU=1.0
# Small timestep while keeping the full sweep runtime manageable.
DT=0.01
RECORD_DT=0.01

# Long warmup + short final record window.
T_WARMUP_TARGET=3000
T_WARMUP_REAL=10000
T_RECORD=200

ICS=(1.0 -1.0 2.0 -2.0)
TAUKS=(3.6 4.13 4.80 4.90)

LC2_TARGET_APPEAR=4.13
LC2_TARGET_DISAPPEAR=4.24
LC2_APPEAR_CUTOFF=3.985
LC2_DISAPPEAR_CUTOFF=4.24

mkdir -p "$OUT_DIR"

echo "Compiling timeseries_phase_target_ic.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O3 -o "$EXE" "$SCRIPT_DIR/timeseries_phase_target_ic.cpp"
echo "Done."

target_tauk_for_ic() {
    local tauk_real="$1"
    local ic="$2"
    python3 - <<PY
x=float("$tauk_real")
ic=float("$ic")
if abs(ic) < 1.5:
    print(f"{x:.6f}")
elif x < float("$LC2_APPEAR_CUTOFF"):
    print(f"{float('$LC2_TARGET_APPEAR'):.6f}")
elif x > float("$LC2_DISAPPEAR_CUTOFF"):
    print(f"{float('$LC2_TARGET_DISAPPEAR'):.6f}")
else:
    print(f"{x:.6f}")
PY
}

run_one() {
    local tauk_real="$1"
    local ic="$2"

    local tauk_target
    tauk_target="$(target_tauk_for_ic "$tauk_real" "$ic")"

    local k_real k_target
    k_real=$(awk -v x="$tauk_real" -v tau="$TAU" 'BEGIN{printf "%.10f", x/tau}')
    k_target=$(awk -v x="$tauk_target" -v tau="$TAU" 'BEGIN{printf "%.10f", x/tau}')

    local tag_real tag_target tag_ic out_file
    tag_real="$(printf '%.6f' "$tauk_real" | tr '-' 'n')"
    tag_target="$(printf '%.6f' "$tauk_target" | tr '-' 'n')"
    tag_ic="$(printf '%.1f' "$ic" | tr '-' 'n')"
    out_file="$OUT_DIR/tauk_${tag_real}_target_${tag_target}_ic_${tag_ic}.tsv"

    if [ -f "$out_file" ]; then
        echo "  using existing tau*k=$tauk_real, ic=$ic, target=$tauk_target"
        return
    fi

    "$EXE" \
        "$TAU" "$k_real" "$k_target" "$ic" "$DT" \
        "$T_WARMUP_TARGET" "$T_WARMUP_REAL" "$T_RECORD" "$RECORD_DT" "$out_file"

    echo "  simulated tau*k=$tauk_real, ic=$ic, target=$tauk_target"
}

echo "Running timeseries jobs ..."
for tauk in "${TAUKS[@]}"; do
    for ic in "${ICS[@]}"; do
        run_one "$tauk" "$ic"
    done
done

echo "Done. Data in: $OUT_DIR"
echo "Next: python3 $SCRIPT_DIR/viz_timeseries_phaseplots.py"
