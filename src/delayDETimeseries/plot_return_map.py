import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re
import sys

def create_return_map(time_series, tau_steps, n=1):
    """Create discrete nth iterate return map: theta(m*tau) vs theta((m+n)*tau)"""
    # Sample at every tau interval and map n delays forward
    # This shows the full phase space structure
    n_points = len(time_series) // tau_steps
    
    theta_m_tau = []
    theta_m_plus_n_tau = []
    
    for i in range(n_points - n):
        theta_m_tau.append(time_series[i * tau_steps])
        theta_m_plus_n_tau.append(time_series[(i + n) * tau_steps])
    
    return np.array(theta_m_tau), np.array(theta_m_plus_n_tau)

def plot_return_map(tau, k, theta0, dt, tmax, n=1):
    """Plot nth iterate return maps for selected noise values"""
    
    script_dir = Path(__file__).parent
    
    # Find the parameter folder (use :g to format floats without trailing zeros)
    folder_pattern = f"tau_{tau:g}_k_{k:g}_theta0_{theta0:g}_dt_{dt:g}_tmax_{tmax:g}"
    folder_path = script_dir / "outputs/SDDETimeseries/long" / folder_pattern
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    # Find all eta values
    files = list(folder_path.glob("eta_*_simNo_*.tsv"))
    if not files:
        print(f"No simulation files found in {folder_path}")
        return
    
    eta_set = set()
    for file in files:
        match = re.search(r'eta_([\d.]+)_simNo', str(file))
        if match:
            eta_set.add(float(match.group(1)))
    
    eta_values = sorted(list(eta_set))

    print(f"Plotting {len(eta_values)} eta values: {eta_values}")
    
    # Create output directory
    output_dir = script_dir / f"plots/return_maps/tau_{tau:g}_k_{k:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine grid size based on number of plots
    n_plots = len(eta_values)
    if n_plots <= 6:
        nrows, ncols = 2, 3
    elif n_plots <= 8:
        nrows, ncols = 2, 4
    elif n_plots <= 9:
        nrows, ncols = 3, 3
    else:
        nrows, ncols = 3, 4
    
    # Create figure with appropriate grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten()
    
    iterate_suffix = "" if n == 1 else f" ({n}th Iterate)"
    fig.suptitle(f'Discrete Return Maps{iterate_suffix}: θ(mτ+{n}τ) vs θ(mτ), Delay DE, τ={tau}, k={k}', 
                 fontsize=14, fontweight='bold')
    
    # Calculate tau in time steps
    tau_steps = int(tau / dt)
    
    # Process each selected eta value
    for subplot_idx, eta in enumerate(eta_values):
        ax = axes[subplot_idx]
        
        # Find all simulations for this eta
        eta_str = "0" if eta == 0.0 else str(eta)
        sim_files = sorted(folder_path.glob(f"eta_{eta_str}_simNo_*.tsv"))
        
        if not sim_files:
            print(f"No simulations found for eta={eta}")
            ax.text(0.5, 0.5, f'No data for η={eta}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"\nProcessing η={eta} ({len(sim_files)} simulations)")
        
        # Collect all return map points from all simulations
        all_theta_t = []
        all_theta_t_plus_ntau = []
        
        for sim_file in sim_files:
            try:
                data = np.loadtxt(sim_file, skiprows=1)
                time = data[:, 0]
                theta = data[:, 1]
                
                # Remove transient
                transient_idx = int(0.5 * len(theta))
                theta_steady = theta[transient_idx:]
                
                # Create nth iterate return map
                theta_t, theta_t_plus_ntau = create_return_map(theta_steady, tau_steps, n)
                all_theta_t.extend(theta_t)
                all_theta_t_plus_ntau.extend(theta_t_plus_ntau)
                
            except Exception as e:
                print(f"  Error loading {sim_file}: {e}")
                continue
        
        if not all_theta_t:
            print(f"No valid data for eta={eta}")
            ax.text(0.5, 0.5, f'Error loading data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Convert to numpy arrays
        all_theta_t = np.array(all_theta_t)
        all_theta_t_plus_ntau = np.array(all_theta_t_plus_ntau)
        
        # Apply modulo 2π to wrap angles
        all_theta_t = np.mod(all_theta_t, 2 * np.pi)
        all_theta_t_plus_ntau = np.mod(all_theta_t_plus_ntau, 2 * np.pi)
        
        # Subsample for plotting if too many points (for clarity)
        max_points = 5000
        if len(all_theta_t) > max_points:
            indices = np.random.choice(len(all_theta_t), max_points, replace=False)
            plot_theta_t = all_theta_t[indices]
            plot_theta_t_plus_ntau = all_theta_t_plus_ntau[indices]
        else:
            plot_theta_t = all_theta_t
            plot_theta_t_plus_ntau = all_theta_t_plus_ntau
        
        # Plot as scatter points (discrete return map)
        ax.scatter(plot_theta_t, plot_theta_t_plus_ntau, s=10, 
                  alpha=0.3, c='blue', edgecolors='none')
        
        # Add diagonal line for reference (fixed point line)
        padding = 0.1
        ax.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', 
               linewidth=1.5, alpha=0.5, label=f'θ(mτ+{n}τ)=θ(mτ)')
        
        # Set fixed axes limits with padding
        ax.set_xlim(-padding, 2*np.pi + padding)
        ax.set_ylim(-padding, 2*np.pi + padding)
                
        ax.set_xlabel('θ(mτ)', fontsize=11)
        ax.set_ylabel(f'θ(mτ+{n}τ)', fontsize=11)
        ax.set_title(f'η = {eta:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide any unused subplots
    for idx in range(len(eta_values), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    iterate_name = "" if n == 1 else f"{n}th_iterate_"
    output_file = output_dir / f"return_map_{iterate_name}tau_{tau:g}_k_{k:g}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()

if __name__ == '__main__':
    tau = 50
    k = 0.08
    theta0 = 1.5708
    dt = 0.1
    tmax = 4000
    
    # Get tau and k from command line arguments if provided
    if len(sys.argv) >= 3:
        tau = float(sys.argv[1])
        k = float(sys.argv[2])
    
    print(f"Generating return maps for tau={tau}, k={k}")
    
    plot_return_map(tau, k, theta0, dt, tmax, n=1)
    plot_return_map(tau, k, theta0, dt, tmax, n=4)
    # plot_return_map(tau, k, theta0, dt, tmax, n=32)
