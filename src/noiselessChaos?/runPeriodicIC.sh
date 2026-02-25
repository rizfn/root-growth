#!/bin/bash

# Parameter sweep for noiseHeunPeriodicIC
# Edit the arrays below to configure the sweep.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Sweep parameters ──────────────────────────────────────────────
TAU_VALUES=(25.0)
K_VALUES=(0.1)
AMPLITUDES=(0.1 1.0)
PERIODS=(1.0 5.0 10.0)

# Fixed
ETA=0.0
DT=0.001
RECORD_DT=0.1
TMAX=4000.0
SIM_NO=0

# ── Compile ───────────────────────────────────────────────────────
echo "Compiling noiseHeunPeriodicIC..."
g++ -fdiagnostics-color=always -std=c++17 -O2 \
    -o "$SCRIPT_DIR/noiseHeunPeriodicIC" "$SCRIPT_DIR/noiseHeunPeriodicIC.cpp"
echo "Done."

# ── Run all combinations in parallel ─────────────────────────────
N_PROCS=$(( $(nproc) - 2 ))
[ $N_PROCS -lt 1 ] && N_PROCS=1

run_one() {
    local tau=$1 k=$2 amp=$3 period=$4
    "$SCRIPT_DIR/noiseHeunPeriodicIC" \
        "$tau" "$k" "$ETA" "$amp" "$period" "$DT" "$TMAX" "$SIM_NO" "$RECORD_DT" \
        > /dev/null 2>&1
}
export -f run_one
export ETA DT RECORD_DT TMAX SIM_NO SCRIPT_DIR

running=0
for tau in "${TAU_VALUES[@]}"; do
for k in "${K_VALUES[@]}"; do
for amp in "${AMPLITUDES[@]}"; do
for period in "${PERIODS[@]}"; do
    run_one "$tau" "$k" "$amp" "$period" &
    running=$(( running + 1 ))
    if [ $running -ge $N_PROCS ]; then
        wait -n
        running=$(( running - 1 ))
    fi
done; done; done; done
wait

echo "Sweep complete."
