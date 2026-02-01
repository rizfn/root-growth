#!/bin/bash

# Fixed parameters
TAU=100.0
K=0.016
THETA0=1.5708
DT=0.1
TMAX=1000.0

# Number of simulations per noise value for ensemble averaging
N_SIMS=20

# Array of noise values to test (logarithmic spacing)
# From 10^-4 to 10^-1, plus eta=0 for reference
NOISE_VALUES=(0.0 0.0001 0.0002 0.0003 0.0005 0.0007 0.001 0.002 0.003 0.005 0.007 0.01 0.02 0.03 0.05 0.07 0.1)

# Get number of processors and keep 2 free
N_PROCS=$(($(nproc) - 2))
if [ $N_PROCS -lt 1 ]; then
    N_PROCS=1
fi

echo "Running simulations with $N_PROCS parallel jobs..."
echo "Testing ${#NOISE_VALUES[@]} noise values with $N_SIMS simulations each"
echo "Total simulations: $((${#NOISE_VALUES[@]} * N_SIMS))"

# Function to run a single simulation
run_simulation() {
    local eta=$1
    local sim_no=$2
    ./noiseEulerMaruyama $TAU $K $eta $THETA0 $DT $TMAX $sim_no > /dev/null 2>&1
}

export -f run_simulation
export TAU K THETA0 DT TMAX

# Run simulations in parallel using GNU parallel if available
if command -v parallel &> /dev/null; then
    # Generate all combinations of eta and sim_no
    for eta in "${NOISE_VALUES[@]}"; do
        for ((sim=0; sim<N_SIMS; sim++)); do
            echo "$eta $sim"
        done
    done | parallel -j $N_PROCS --col-sep ' ' run_simulation {1} {2}
else
    # Fallback: use background jobs with job control
    running_jobs=0
    total_jobs=$((${#NOISE_VALUES[@]} * N_SIMS))
    completed=0
    
    for eta in "${NOISE_VALUES[@]}"; do
        for ((sim=0; sim<N_SIMS; sim++)); do
            run_simulation $eta $sim &
            running_jobs=$((running_jobs + 1))
            
            # Wait if we've reached the max number of parallel jobs
            if [ $running_jobs -ge $N_PROCS ]; then
                wait -n  # Wait for any job to finish
                running_jobs=$((running_jobs - 1))
                completed=$((completed + 1))
                echo -ne "\rCompleted: $completed/$total_jobs"
            fi
        done
    done
    wait  # Wait for all remaining jobs to finish
    echo -e "\rCompleted: $total_jobs/$total_jobs"
fi

echo ""
echo "All simulations complete!"
echo "Results saved in outputs/SDDETimeseries/"
