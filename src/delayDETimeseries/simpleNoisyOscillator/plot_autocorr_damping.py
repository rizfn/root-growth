import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from pathlib import Path
import glob
import re

def compute_autocorrelation(time_series, max_lag=500):
    """Compute normalized autocorrelation function with unbiased normalization"""
    n = len(time_series)
    mean = np.mean(time_series)
    var = np.var(time_series)
    
    if var == 0:
        return np.ones(max_lag)
    
    time_series_normalized = time_series - mean
    autocorr = np.correlate(time_series_normalized, time_series_normalized, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag]
    
    # Unbiased normalization: divide by number of overlapping points at each lag
    # This prevents artificial decay due to fewer samples at large lags
    for lag in range(len(autocorr)):
        autocorr[lag] = autocorr[lag] / (var * (n - lag))
    
    return autocorr

def fit_damped_oscillation(lag_times, autocorr):
    """Fit autocorrelation to damped oscillation: A*exp(-lambda*t)*cos(omega*t + phi)"""
    # Find dominant frequency from FFT
    fft_vals = np.fft.fft(autocorr)
    freqs = np.fft.fftfreq(len(autocorr), lag_times[1] - lag_times[0])
    peak_freq = abs(freqs[np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1])
    omega_guess = 2 * np.pi * peak_freq
    
    # Estimate damping from envelope
    envelope = np.abs(signal.hilbert(autocorr))
    try:
        valid_idx = envelope > 0.01
        if np.sum(valid_idx) > 10:
            popt_env, _ = curve_fit(lambda t, A, lam: A * np.exp(-lam * t), 
                                   lag_times[valid_idx], envelope[valid_idx],
                                   p0=[1.0, 0.01], maxfev=2000)
            lambda_guess = popt_env[1]
        else:
            lambda_guess = 0.01
    except:
        lambda_guess = 0.01
    
    # Fit full damped oscillation
    def damped_osc(t, A, lam, omega, phi):
        return A * np.exp(-lam * t) * np.cos(omega * t + phi)
    
    try:
        popt, _ = curve_fit(damped_osc, lag_times, autocorr,
                           p0=[1.0, lambda_guess, omega_guess, 0.0],
                           maxfev=5000)
        return popt  # [A, lambda, omega, phi]
    except:
        return None

def plot_autocorr_with_envelope(k, gamma, theta0, omega0, dt, tmax):
    """Plot autocorrelation and envelope for selected noise values"""
    
    script_dir = Path(__file__).parent
    
    # Find the parameter folder
    folder_pattern = f"k_{k}_gamma_{gamma}_theta0_{theta0}_omega0_{omega0}_dt_{dt}_tmax_{tmax}"
    folder_path = script_dir / "outputs/SimpleOscillator" / folder_pattern
    
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
    print(f"Found {len(eta_values)} eta values")
    
    # Select 6 noise values evenly spaced
    if len(eta_values) <= 6:
        selected_etas = eta_values
    else:
        indices = np.linspace(0, len(eta_values)-1, 6, dtype=int)
        selected_etas = [eta_values[i] for i in indices]
    
    print(f"Selected eta values: {selected_etas}")
    
    # Create output directory
    output_dir = script_dir / "plots/autocorrelation_damping"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 6 subplots (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    if gamma == 0.0:
        fig.suptitle(f'Autocorrelation Damping: Undamped Harmonic Oscillator, ω₀={k}', 
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Autocorrelation Damping: Damped Harmonic Oscillator, k={k}, γ={gamma}', 
                     fontsize=14, fontweight='bold')
    
    # Process each selected eta value
    for subplot_idx, eta in enumerate(selected_etas):
        ax = axes[subplot_idx]
        
        # Find all simulations for this eta and ensemble average
        # Format eta to match file naming (0.0 -> "0", 0.001 -> "0.001")
        eta_str = str(int(eta)) if eta == int(eta) else str(eta)
        sim_files = sorted(folder_path.glob(f"eta_{eta_str}_simNo_*.tsv"))
        
        if not sim_files:
            print(f"No simulations found for eta={eta}")
            ax.text(0.5, 0.5, f'No data for η={eta}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"\nProcessing η={eta} ({len(sim_files)} simulations)")
        
        # Load and process all simulations
        all_autocorr = []
        all_params = []
        
        for sim_file in sim_files:
            try:
                data = np.loadtxt(sim_file, skiprows=1)
                time = data[:, 0]
                theta = data[:, 1]
                
                # Remove transient
                transient_idx = int(0.2 * len(theta))
                theta_steady = theta[transient_idx:]
                
                # Compute autocorrelation
                max_lag_time = 500.0
                max_lag = min(int(max_lag_time / dt), len(theta_steady)//2)
                autocorr = compute_autocorrelation(theta_steady, max_lag)
                all_autocorr.append(autocorr)
                
                # Fit parameters
                if eta > 0:
                    lag_times = np.arange(max_lag) * dt
                    fit_params = fit_damped_oscillation(lag_times, autocorr)
                    if fit_params is not None:
                        all_params.append(fit_params)
                
            except Exception as e:
                print(f"  Error loading {sim_file}: {e}")
                continue
        
        if not all_autocorr:
            print(f"No valid data for eta={eta}")
            ax.text(0.5, 0.5, f'Error loading data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Ensemble average
        autocorr_mean = np.mean(all_autocorr, axis=0)
        lag_times = np.arange(len(autocorr_mean)) * dt
        
        # Plot autocorrelation
        ax.plot(lag_times, autocorr_mean, 'b-', linewidth=2, label='Autocorrelation', alpha=0.8)
        
        # Plot envelope if we have fit parameters
        if all_params:
            params_mean = np.array(all_params).mean(axis=0)
            A, lam, omega, phi = params_mean
            
            # Plot upper and lower envelopes
            envelope_upper = A * np.exp(-lam * lag_times)
            envelope_lower = -A * np.exp(-lam * lag_times)
            
            ax.plot(lag_times, envelope_upper, 'r--', linewidth=2, 
                   label=f'Envelope (λ={lam:.4f})', alpha=0.8)
            ax.plot(lag_times, envelope_lower, 'r--', linewidth=2, alpha=0.8)
            
            # Add text with fit parameters
            textstr = f'η = {eta:.4f}\nλ = {lam:.4f}\nω = {omega:.4f}\nA = {A:.4f}'
            ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            textstr = f'η = {eta:.4f}\n(no damping)'
            ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Lag Time (s)', fontsize=11)
        ax.set_ylabel('Autocorrelation', fontsize=11)
        ax.set_title(f'η = {eta:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"autocorr_damping_k_{k}_gamma_{gamma}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    

if __name__ == '__main__':
    # Parameters (should match the bash script)
    k = 0.5
    gamma = 0
    theta0 = 1.5708
    omega0 = 0
    dt = 0.1
    tmax = 1000
    
    plot_autocorr_with_envelope(k, gamma, theta0, omega0, dt, tmax)
