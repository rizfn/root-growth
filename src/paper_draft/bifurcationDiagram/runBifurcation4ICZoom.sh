#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXE="$SCRIPT_DIR/bifurcation_target_ic"
OUT_ROOT="$SCRIPT_DIR/outputs/bifurcation4IC_zoom"

TAU=25
DT=0.01
RECORD_DT=0.05

T_WARMUP_TARGET=3000
T_WARMUP_REAL=12000
T_MEASURE=40000

# Same branch-target guidance as the full scan script.
LC2_TARGET_APPEAR=4.13
LC2_TARGET_DISAPPEAR=4.24
LC2_APPEAR_CUTOFF=3.985
LC2_DISAPPEAR_CUTOFF=4.24

LC2_MIN=4.100
LC2_MAX=4.250
LC2_POINTS=200

INT_MIN=4.800
INT_MAX=5.000
INT_POINTS=200

mkdir -p "$OUT_ROOT"

echo "Compiling bifurcation_target_ic.cpp ..."
g++ -fdiagnostics-color=always -std=c++17 -O2 -o "$EXE" "$SCRIPT_DIR/bifurcation_target_ic.cpp"
echo "Done."

N_PROCS=$(($(nproc) - 2))
if [ "$N_PROCS" -lt 1 ]; then N_PROCS=1; fi
echo "Using $N_PROCS parallel jobs (keeping 2 CPUs free)."

run_one() {
    local out_dir="$1"
    local k_real="$2"
    local tauk_target="$3"
    local theta0="$4"
    local branch="$5"

    local k_target
    k_target=$(awk -v tkt="$tauk_target" -v tau="$TAU" 'BEGIN{printf "%.10f", tkt/tau}')

    local tauk_real
    tauk_real=$(awk -v k="$k_real" -v tau="$TAU" 'BEGIN{printf "%.6f", k*tau}')

    local tag_real tag_target tag_ic out_file
    tag_real="$(printf '%.6f' "$tauk_real" | tr '-' 'n')"
    tag_target="$(printf '%.6f' "$tauk_target" | tr '-' 'n')"
    tag_ic="$(printf '%.1f' "$theta0" | tr '-' 'n')"

    out_file="$out_dir/${branch}_tauk_${tag_real}_target_${tag_target}_ic_${tag_ic}.tsv"

    "$EXE" \
        "$TAU" "$k_real" "$k_target" "$theta0" "$DT" \
        "$T_WARMUP_TARGET" "$T_WARMUP_REAL" "$T_MEASURE" "$RECORD_DT" "$out_file"
}

export -f run_one
export EXE TAU DT T_WARMUP_TARGET T_WARMUP_REAL T_MEASURE RECORD_DT

run_interval() {
    local interval_name="$1"
    local tauk_min="$2"
    local tauk_max="$3"
    local n_points="$4"

    local out_dir="$OUT_ROOT/$interval_name"
    mkdir -p "$out_dir"

    read -r -a K_VALS <<< "$(python3 - <<PY
import numpy as np
vals = np.linspace($tauk_min, $tauk_max, $n_points) / $TAU
print(' '.join(f'{v:.10f}' for v in vals))
PY
)"

    local pairs=()
    for k in "${K_VALS[@]}"; do
        local tauk
        tauk=$(awk -v kv="$k" -v tau="$TAU" 'BEGIN{printf "%.6f", kv*tau}')

        pairs+=("$out_dir $k $tauk 1.0 p4_pos")
        pairs+=("$out_dir $k $tauk -1.0 p4_neg")

        local tgt_lc2
        tgt_lc2=$(python3 - <<PY
x=$tauk
if x < $LC2_APPEAR_CUTOFF:
    print(f"{$LC2_TARGET_APPEAR:.6f}")
elif x > $LC2_DISAPPEAR_CUTOFF:
    print(f"{$LC2_TARGET_DISAPPEAR:.6f}")
else:
    print(f"{x:.6f}")
PY
)
        pairs+=("$out_dir $k $tgt_lc2 2.0 lc2_pos")
        pairs+=("$out_dir $k $tgt_lc2 -2.0 lc2_neg")
    done

    echo ""
    echo "=== $interval_name: tau*k in [$tauk_min, $tauk_max], ${#K_VALS[@]} points ==="
    echo "Running ${#pairs[@]} jobs ..."

    if command -v parallel >/dev/null 2>&1; then
        printf '%s\n' "${pairs[@]}" | parallel -j "$N_PROCS" --colsep ' ' run_one {1} {2} {3} {4} {5}
    else
        local running=0
        for pair in "${pairs[@]}"; do
            local out_dir_i k_real_i tauk_tgt_i theta0_i branch_i
            read -r out_dir_i k_real_i tauk_tgt_i theta0_i branch_i <<< "$pair"
            run_one "$out_dir_i" "$k_real_i" "$tauk_tgt_i" "$theta0_i" "$branch_i" &
            running=$((running + 1))
            if [ "$running" -ge "$N_PROCS" ]; then
                wait -n
                running=$((running - 1))
            fi
        done
        wait
    fi
}

run_interval "lc2" "$LC2_MIN" "$LC2_MAX" "$LC2_POINTS"
run_interval "intermittency" "$INT_MIN" "$INT_MAX" "$INT_POINTS"

echo ""
echo "Done. Outputs:"
echo "  $OUT_ROOT/lc2"
echo "  $OUT_ROOT/intermittency"
echo "Next: python3 $SCRIPT_DIR/viz_bifurcation_4ic_zoom.py"
