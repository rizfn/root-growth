#!/bin/bash
set -euo pipefail

# Cover figure: the chaotic attractor on the LC2 branch at tau*k = 4.235
# (largest Lyapunov exponent ~0.21/tau), reached from the constant history
# theta(s) = 2 on [-tau, 0].  Same integrator as the paper figures: explicit
# trapezoidal rule, dt = 1e-3, tau/dt an integer.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/../../../root-env/bin/python"
EXE="$SCRIPT_DIR/attractorRun"
OUT_DIR="$SCRIPT_DIR/outputs"

TAUK=4.235
THETA0=2.0
DT=0.001
T_WARMUP=3000
T_MEASURE=100000
RECORD_DT=0.002

mkdir -p "$OUT_DIR"
g++ -O3 -march=native -std=c++17 -o "$EXE" "$SCRIPT_DIR/attractorRun.cpp"

DATA="$OUT_DIR/tauk_${TAUK}_theta0_2_tmeasure_${T_MEASURE}_rdt_${RECORD_DT}.bin"
if [ ! -f "$DATA" ]; then
    "$EXE" "$TAUK" "$THETA0" "$DT" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT" "$DATA"
fi

COMMON=(--input "$DATA" --tauk "$TAUK" --record-dt "$RECORD_DT" --lags 0 1 2 --az 60 --el 35 --dpi 300)

# Portrait, with the top 20% left free for the masthead.
for palette in ember root; do
    "$PYTHON" "$SCRIPT_DIR/viz_cover_attractor.py" "${COMMON[@]}" --palette "$palette"
done
# Square, attractor centred.
"$PYTHON" "$SCRIPT_DIR/viz_cover_attractor.py" "${COMMON[@]}" --palette ember \
    --width-in 8 --height-in 8 --top 0.05 --bottom 0.05

# With real seedlings: cut them out of the plate photographs, then compose.
if [ ! -f "$OUT_DIR/root_crops.npz" ]; then
    "$PYTHON" "$SCRIPT_DIR/extract_roots.py"
fi
"$PYTHON" "$SCRIPT_DIR/compose_cover.py" --dpi 300
