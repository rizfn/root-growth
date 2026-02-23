import matplotlib.pyplot as plt
import numpy as np
import os

def main(tau, k, eta, theta0, dt, tmax):
    # Load data
    script_dir = os.path.dirname(__file__)
    data_file = f"{script_dir}/outputs/SDDETimeseries/long/tau_{tau}_k_{k}_theta0_{theta0}_dt_{dt}_tmax_{tmax}/eta_{eta}_simNo_0.tsv"
    data = np.loadtxt(data_file, skiprows=1)
    
    # Calculate tau in timesteps
    tau_steps = int(tau / dt)
    
    # Sample at tau intervals
    time_full = data[:, 0]
    theta_full = data[:, 1]
    
    # Get discrete samples at multiples of tau
    time_discrete = time_full[::tau_steps]
    theta_discrete = theta_full[::tau_steps]
    
    # Plot
    plt.figure(figsize=(12, 6))
    # Plot underlying trajectory faintly
    plt.plot(time_full, theta_full, linewidth=0.5, alpha=0.4, color='gray', label='Full trajectory')
    # Plot discrete samples with markers
    plt.plot(time_discrete, theta_discrete, 'o-', linewidth=1.5, markersize=4, 
             alpha=0.8, color='blue', label=f'Sampled at τ intervals')
    
    plt.xlabel('Time')
    plt.ylabel('θ (radians)')
    plt.title(f'Discrete SDDE (τ={tau}): k={k}, η={eta}, θ₀={theta0:.4f}')
    plt.xlim(3*tmax/4, tmax)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save
    output_dir = f"{script_dir}/plots/SDDETimeseries/discrete"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/tau_{tau}_k_{k}_eta_{eta}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    tau = 20
    k = 0.19
    eta = 0
    theta0 = 1.5708
    dt = 0.01
    tmax = 4000
    main(tau, k, eta, theta0, dt, tmax)
