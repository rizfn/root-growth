#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$SCRIPT_DIR/timeseries_lnp1"

g++ -O2 -std=c++17 -march=native -o "$BIN" "$SCRIPT_DIR/timeseries_lnp1.cpp"
echo "Compiled: $BIN"

T_WARMUP=2000
T_MEASURE=3000
THETA0=1.0
RECORD_DT=0.1
NJOBS=8

pids=()
for k in $(seq 0.5 0.1 5.0); do
    "$BIN" "$k" "$THETA0" "$T_WARMUP" "$T_MEASURE" "$RECORD_DT" &
    pids+=($!)
    if [[ ${#pids[@]} -ge $NJOBS ]]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

echo "Done. Outputs in $SCRIPT_DIR/outputs/lnp1/"
