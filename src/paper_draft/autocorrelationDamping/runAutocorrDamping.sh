#!/usr/bin/env bash
set -euo pipefail

# Parallel tau=1 noise sweep for autocorrelation damping figures.
#
# Usage:
#   bash runAutocorrDamping.sh
#   bash runAutocorrDamping.sh <n_sims>
#
# The solver writes to outputs/SDDETimeseries/tau_k_raster/ and the plotting
# script reads the same folders to build the two-panel damping figures.

N_SIMS="${1:-10}"

TAU=1
THETA0=1.5708
DT=0.001
RECORD_DT=0.1
TMAX=1000

K_VALUES=(1.6 2 4 4.7)
MU_VALUES=(0.0 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.002 0.003 0.005 0.007 0.01 0.02 0.03 0.05 0.07 0.1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Compiling noiseHeun.cpp..."
g++ -std=c++17 -O2 -g "$SCRIPT_DIR/noiseHeun.cpp" -o "$SCRIPT_DIR/noiseHeun"

N_PROCS=$(($(nproc) - 2))
if [ "$N_PROCS" -lt 1 ]; then
  N_PROCS=1
fi

run_one() {
  local k="$1"
  local mu="$2"
  local sim_no="$3"
  "$SCRIPT_DIR/noiseHeun" "$TAU" "$k" "$mu" "$THETA0" "$DT" "$TMAX" "$sim_no" "$RECORD_DT"
}

export -f run_one
export SCRIPT_DIR TAU THETA0 DT TMAX RECORD_DT

PAIRS=()
for k in "${K_VALUES[@]}"; do
  for mu in "${MU_VALUES[@]}"; do
    for ((sim_no=0; sim_no<N_SIMS; sim_no++)); do
      PAIRS+=("$k $mu $sim_no")
    done
  done
done

total_jobs=${#PAIRS[@]}
echo "Running $total_jobs simulations across ${#K_VALUES[@]} k-values, ${#MU_VALUES[@]} noise values, and $N_SIMS simulations per pair."

if command -v parallel >/dev/null 2>&1; then
  printf '%s\n' "${PAIRS[@]}" | parallel -j "$N_PROCS" --colsep ' ' run_one {1} {2} {3}
else
  running=0
  for pair in "${PAIRS[@]}"; do
    read -r k mu sim_no <<< "$pair"
    run_one "$k" "$mu" "$sim_no" &
    running=$((running + 1))
    if [ "$running" -ge "$N_PROCS" ]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
fi

echo "Sweep complete."