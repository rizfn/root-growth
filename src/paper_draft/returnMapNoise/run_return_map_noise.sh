#!/usr/bin/env bash
set -euo pipefail

# One-command pipeline for return-map noise study (tau fixed to 1 in noiseHeun.cpp)
# Usage:
#   ./run_return_map_noise.sh
#   ./run_return_map_noise.sh <k> <tmax> <n_sims>
# Example:
#   ./run_return_map_noise.sh 2 4000 1

K="${1:-3}"
TMAX="${2:-4000}"
N_SIMS="${3:-1}"

TAU=1
THETA0=1.5708
DT=0.001
RECORD_DT=0.01

NOISE_VALUES=(0.0 0.001 0.01 0.1)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building noiseHeun.cpp..."
g++ -std=c++17 -O2 -g noiseHeun.cpp -o noiseHeun

echo "Running eta sweep: k=$K, tau=$TAU, dt=$DT, record_dt=$RECORD_DT, tmax=$TMAX, sims=$N_SIMS"

MAX_JOBS=$(($(nproc) - 2))
if [ "$MAX_JOBS" -lt 1 ]; then
  MAX_JOBS=1
fi

running_jobs=0
total_jobs=$((${#NOISE_VALUES[@]} * N_SIMS))
completed=0

for eta in "${NOISE_VALUES[@]}"; do
  for ((sim=0; sim<N_SIMS; sim++)); do
    ./noiseHeun "$TAU" "$K" "$eta" "$THETA0" "$DT" "$TMAX" "$sim" "$RECORD_DT" > /dev/null &
    running_jobs=$((running_jobs + 1))

    if [ "$running_jobs" -ge "$MAX_JOBS" ]; then
      wait -n
      running_jobs=$((running_jobs - 1))
      completed=$((completed + 1))
      echo -ne "\rCompleted: $completed/$total_jobs"
    fi
  done
done

while [ "$running_jobs" -gt 0 ]; do
  wait -n
  running_jobs=$((running_jobs - 1))
  completed=$((completed + 1))
  echo -ne "\rCompleted: $completed/$total_jobs"
done
echo

echo "Generating return-map plots..."
python "$SCRIPT_DIR/viz_return_map_noise.py" "$TAU" "$K" "$THETA0" "$RECORD_DT" "$TMAX" 1

echo "Done. See plots in: $SCRIPT_DIR/plots"
