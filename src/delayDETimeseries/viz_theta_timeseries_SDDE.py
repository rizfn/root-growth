import matplotlib.pyplot as plt
import numpy as np
import os

def main(tau, k, eta, theta0, dt, tmax):
    # Load data
    script_dir = os.path.dirname(__file__)
    data_file = f"{script_dir}/outputs/SDDETimeseries/tau_{tau}_k_{k}_eta_{eta}_theta0_{theta0}_dt_{dt}_tmax_{tmax}.tsv"
    data = np.loadtxt(data_file, skiprows=1)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data[:, 0], data[:, 1], linewidth=0.8, alpha=0.8)
    plt.xlabel('Time')
    plt.ylabel('θ (radians)')
    plt.title(f'SDDE: τ={tau}, k={k}, η={eta}, θ₀={theta0:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Save
    output_dir = f"{script_dir}/plots/SDDETimeseries"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/tau_{tau}_k_{k}_eta_{eta}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    tau = 20
    k = 0.08
    eta = 0
    theta0 = 1.5708
    dt = 0.1
    tmax = 1000
    main(tau, k, eta, theta0, dt, tmax)