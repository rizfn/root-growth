import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import os
from pathlib import Path
import glob
import re

def compute_psd(time_series, dt):
    """Compute Power Spectral Density using Welch's method"""
    fs = 1.0 / dt  # Sampling frequency
    frequencies, psd = signal.welch(time_series, fs=fs, nperseg=min(len(time_series)//4, 1024))
    return frequencies, psd

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
    # Initial guess for parameters
    # Find dominant frequency from FFT
    fft_vals = np.fft.fft(autocorr)
    freqs = np.fft.fftfreq(len(autocorr), lag_times[1] - lag_times[0])
    peak_freq = abs(freqs[np.argmax(np.abs(fft_vals[1:len(fft_vals)//2])) + 1])
    omega_guess = 2 * np.pi * peak_freq
    
    # Estimate damping from envelope
    envelope = np.abs(signal.hilbert(autocorr))
    try:
        # Fit exponential to envelope
        valid_idx = envelope > 0.01  # Avoid numerical issues
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

def analyze_timeseries_ensemble(k, gamma, theta0, omega0, dt, tmax):
    """Analyze multiple timeseries with different noise levels using ensemble averaging"""
    
    script_dir = Path(__file__).parent
    
    # Find the parameter folder
    folder_pattern = f"k_{k}_gamma_{gamma}_theta0_{theta0}_omega0_{omega0}_dt_{dt}_tmax_{tmax}"
    folder_path = script_dir / "outputs/SimpleOscillator" / folder_pattern
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return
    
    # Find all eta values by looking at files in the folder
    files = list(folder_path.glob("eta_*_simNo_*.tsv"))
    if not files:
        print(f"No simulation files found in {folder_path}")
        return
    
    # Extract unique eta values
    eta_set = set()
    for file in files:
        match = re.search(r'eta_([\d.]+)_simNo', str(file))
        if match:
            eta_set.add(float(match.group(1)))
    
    eta_values = sorted(list(eta_set))
    print(f"Found {len(eta_values)} eta values: {eta_values}")
    
    # Separate eta=0 from non-zero values for plotting
    eta_zero_present = 0.0 in eta_values
    eta_values_nonzero = [eta for eta in eta_values if eta > 0]
    
    # Create output directory
    output_dir = script_dir / "plots/noise_analysis_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    if gamma == 0.0:
        fig.suptitle(f'Ensemble Noise Analysis: Undamped Harmonic Oscillator, ω₀={k}', fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Ensemble Noise Analysis: Damped Harmonic Oscillator, k={k}, γ={gamma}', fontsize=14, fontweight='bold')
    
    # Colors for different noise levels (including eta=0 if present)
    colors = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    # Store results for ensemble statistics (only non-zero eta for log plots)
    damping_rates_mean = []
    damping_rates_std = []
    eta_nonzero_with_data = []
    variance_mean = []
    variance_std = []
    variance_eta = []  # Separate list for eta values with variance data
    
    for idx, eta in enumerate(eta_values):
        print(f"Processing eta={eta}...")
        
        # Find all simulations for this eta
        sim_files = sorted(folder_path.glob(f"eta_{eta}_simNo_*.tsv"))
        
        if not sim_files:
            print(f"  No simulations found for eta={eta}")
            continue
        
        print(f"  Found {len(sim_files)} simulations")
        
        # Load all simulations
        all_theta_steady = []
        all_time_steady = []
        all_autocorr = []
        all_psd = []
        all_variances = []
        all_damping = []
        
        for sim_file in sim_files:
            try:
                data = np.loadtxt(sim_file, skiprows=1)
                time = data[:, 0]
                theta = data[:, 1]
                
                # Remove transient (first 20% of data)
                transient_idx = int(0.2 * len(theta))
                theta_steady = theta[transient_idx:]
                time_steady = time[transient_idx:]
                
                all_theta_steady.append(theta_steady)
                all_time_steady.append(time_steady)
                
                # Compute autocorrelation for this simulation
                max_lag_time = 240.0
                max_lag = min(int(max_lag_time / dt), len(theta_steady)//2)
                autocorr = compute_autocorrelation(theta_steady, max_lag)
                all_autocorr.append(autocorr)
                
                # Compute PSD for this simulation
                frequencies, psd = compute_psd(theta_steady, dt)
                all_psd.append((frequencies, psd))
                
                # Compute variance
                all_variances.append(np.var(theta_steady))
                
                # Fit damping for this simulation
                if eta > 0:
                    lag_times = np.arange(max_lag) * dt
                    fit_params = fit_damped_oscillation(lag_times, autocorr)
                    if fit_params is not None:
                        A, lam, omega, phi = fit_params
                        all_damping.append(lam)
                        
            except Exception as e:
                print(f"  Error loading {sim_file}: {e}")
                continue
        
        if not all_theta_steady:
            print(f"  No valid data for eta={eta}")
            continue
        
        # Ensemble average autocorrelation
        autocorr_mean = np.mean(all_autocorr, axis=0)
        autocorr_std = np.std(all_autocorr, axis=0)
        lag_times = np.arange(len(autocorr_mean)) * dt
        
        # Ensemble average PSD (all should have same frequencies)
        psd_values = np.array([psd for _, psd in all_psd])
        psd_mean = np.mean(psd_values, axis=0)
        frequencies = all_psd[0][0]
        
        # Plot 1: Sample time series (one representative)
        if idx % max(1, len(eta_values)//3) == 0:
            axes[0, 0].plot(all_time_steady[0][:2000], all_theta_steady[0][:2000], 
                          alpha=0.7, linewidth=0.8, label=f'η={eta}', color=colors[idx])
        
        # Plot 2: Ensemble-averaged PSD
        axes[0, 1].loglog(frequencies[1:], psd_mean[1:], alpha=0.7, linewidth=1.5, 
                         label=f'η={eta}', color=colors[idx])
        
        # Plot 3: Ensemble-averaged autocorrelation with error band
        axes[1, 0].plot(lag_times, autocorr_mean, alpha=0.8, linewidth=1.5, 
                       label=f'η={eta}', color=colors[idx])
        axes[1, 0].fill_between(lag_times, autocorr_mean - autocorr_std, 
                               autocorr_mean + autocorr_std, alpha=0.2, color=colors[idx])
        
        # Plot 4: Variance with error bars
        var_mean = np.mean(all_variances)
        var_std = np.std(all_variances) / np.sqrt(len(all_variances))  # Standard error
        
        # Plot eta=0 separately (off the log scale)
        if eta == 0.0:
            # Add text annotation for eta=0
            axes[1, 1].text(0.05, 0.95, f'η=0: σ²={var_mean:.4f}±{var_std:.4f}', 
                          transform=axes[1, 1].transAxes, fontsize=10,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[1, 1].errorbar(eta, var_mean, yerr=var_std, fmt='o', markersize=8, 
                               color=colors[idx], capsize=5, zorder=3)
            variance_mean.append(var_mean)
            variance_std.append(var_std)
            variance_eta.append(eta)
        
        # Store damping statistics (only for non-zero noise)
        if eta > 0 and all_damping:
            damp_mean = np.mean(all_damping)
            damp_std = np.std(all_damping) / np.sqrt(len(all_damping))  # Standard error
            damping_rates_mean.append(damp_mean)
            damping_rates_std.append(damp_std)
            eta_nonzero_with_data.append(eta)
    
    # Format subplot 1: Time series
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('θ (radians)')
    axes[0, 0].set_title('Sample Time Series')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Format subplot 2: PSD
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Ensemble-Averaged PSD')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, alpha=0.3, which='both')
    
    # Format subplot 3: Autocorrelation
    axes[1, 0].set_xlabel('Lag Time')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_title('Ensemble-Averaged Autocorrelation')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Format subplot 4: Variance vs noise
    axes[1, 1].set_xlabel('Noise Strength (η)')
    axes[1, 1].set_ylabel('Variance of θ')
    axes[1, 1].set_title('Variance vs Noise Strength')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    # Format subplot 5: Damping rate vs noise with error bars
    if len(damping_rates_mean) > 0:
        axes[0, 2].errorbar(eta_nonzero_with_data, damping_rates_mean, yerr=damping_rates_std,
                           fmt='o', markersize=8, color='darkred', capsize=5, zorder=3)
            
    axes[0, 2].set_xlabel('Noise Strength (η)')
    axes[0, 2].set_ylabel('Damping Rate (λ)')
    axes[0, 2].set_title('Autocorrelation Damping Rate')
    axes[0, 2].grid(True, alpha=0.3, which='both')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_yscale('log')
    
    # Format subplot 6: Correlation time vs noise with error bars
    if len(damping_rates_mean) > 0:
        corr_times_mean = [1/lam if lam > 0 else np.nan for lam in damping_rates_mean]
        # Error propagation: if lambda = mean +/- std, then tau_c = 1/lambda
        # delta(tau_c) = |d(1/lambda)/d(lambda)| * delta(lambda) = std/lambda^2
        corr_times_std = [std/(lam**2) if lam > 0 else np.nan 
                         for lam, std in zip(damping_rates_mean, damping_rates_std)]
        axes[1, 2].errorbar(eta_nonzero_with_data, corr_times_mean, yerr=corr_times_std,
                           fmt='o', markersize=8, color='darkorange', capsize=5, zorder=3)
                
    axes[1, 2].set_xlabel('Noise Strength (η)')
    axes[1, 2].set_ylabel('Correlation Time (τ_c)')
    axes[1, 2].set_title('Correlation Time')
    axes[1, 2].grid(True, alpha=0.3, which='both')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"noise_analysis_ensemble_k_{k}_gamma_{gamma}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nAnalysis saved to: {output_file}")
    
    plt.close()

if __name__ == '__main__':
    # Parameters (should match the bash script)
    k = 0.5
    gamma = 0
    theta0 = 1.5708
    omega0 = 0
    dt = 0.1
    tmax = 1000
    
    analyze_timeseries_ensemble(k, gamma, theta0, omega0, dt, tmax)
