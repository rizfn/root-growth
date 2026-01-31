#!/bin/bash

# Fixed parameters
TAU=5.0
K=0.32
THETA0=1.5708
DT=0.1
TMAX=10000.0

# Array of noise values to test (logarithmic spacing)
# From 10^-4 to 10^-1, plus eta=0 for reference
NOISE_VALUES=(0.0 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.002 0.003 0.005 0.007 0.01 0.02 0.03 0.05 0.07 0.1)

# Get number of processors and keep 2 free
N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then
    N_PROCS=1
fi

echo "Running simulations with $N_PROCS parallel jobs..."
echo "Testing ${#NOISE_VALUES[@]} different noise values"

# Function to run a single simulation
run_simulation() {
    local eta=$1
    echo "Running: eta=$eta"
    ./noiseEulerMaruyama $TAU $K $eta $THETA0 $DT $TMAX
}

export -f run_simulation
export TAU K THETA0 DT TMAX

# Run simulations in parallel using GNU parallel if available
if command -v parallel &> /dev/null; then
    printf "%s\n" "${NOISE_VALUES[@]}" | parallel -j $N_PROCS run_simulation {}
else
    # Fallback: use background jobs with job control
    running_jobs=0
    for eta in "${NOISE_VALUES[@]}"; do
        run_simulation $eta &
        running_jobs=$((running_jobs + 1))
        
        # Wait if we've reached the max number of parallel jobs
        if [ $running_jobs -ge $N_PROCS ]; then
            wait -n  # Wait for any job to finish
            running_jobs=$((running_jobs - 1))
        fi
    done
    wait  # Wait for all remaining jobs to finish
fi

echo ""
echo "All simulations complete!"
echo "Results saved in outputs/delayDE/timeseries/"
