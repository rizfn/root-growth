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

def analyze_timeseries(tau, k, theta0, dt, tmax):
    """Analyze multiple timeseries with different noise levels"""
    
    script_dir = Path(__file__).parent
    
    # Find all matching files using glob
    pattern = str(script_dir / f"outputs/SDDETimeseries/tau_{tau}_k_{k}_eta_*_theta0_{theta0}_dt_{dt}_tmax_{tmax}.tsv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Extract eta values and sort by eta (not by filename)
    file_eta_pairs = []
    for file in files:
        match = re.search(r'eta_([\d.]+)_', file)
        if match:
            eta = float(match.group(1))
            file_eta_pairs.append((eta, file))
    
    if not file_eta_pairs:
        print("Could not extract eta values from filenames")
        return
    
    # Sort by eta value
    file_eta_pairs.sort(key=lambda x: x[0])
    eta_values = [eta for eta, _ in file_eta_pairs]
    files = [file for _, file in file_eta_pairs]
    
    print(f"Found {len(eta_values)} files with eta values: {eta_values}")
    
    # Create output directory
    output_dir = script_dir / "plots/noise_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Noise Effect Analysis: τ={tau}, k={k}', fontsize=14, fontweight='bold')
    
    # Colors for different noise levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(eta_values)))
    
    # Store damping rates for analysis
    damping_rates = []
    eta_nonzero = []
    
    for idx, (eta, data_file) in enumerate(zip(eta_values, files)):
        # Load data
        if not Path(data_file).exists():
            print(f"Warning: File not found: {data_file}")
            continue
        
        data = np.loadtxt(data_file, skiprows=1)
        time = data[:, 0]
        theta = data[:, 1]
        
        # Remove transient (first 20% of data)
        transient_idx = int(0.2 * len(theta))
        theta_steady = theta[transient_idx:]
        time_steady = time[transient_idx:]
        
        # 1. Time series (just a few for visualization)
        if idx % max(1, len(eta_values)//3) == 0:  # Plot only a few to avoid clutter
            axes[0, 0].plot(time_steady[:2000], theta_steady[:2000], 
                          alpha=0.7, linewidth=0.8, label=f'η={eta}', color=colors[idx])
        
        # 2. Power Spectral Density
        frequencies, psd = compute_psd(theta_steady, dt)
        axes[0, 1].loglog(frequencies[1:], psd[1:], alpha=0.7, linewidth=1.5, 
                         label=f'η={eta}', color=colors[idx])
        
        # 3. Autocorrelation: period is ~120, so show autocorrelation up to that time
        max_lag_time = 240.0
        max_lag = min(int(max_lag_time / dt), len(theta_steady)//2)
        autocorr = compute_autocorrelation(theta_steady, max_lag)
        lag_times = np.arange(max_lag) * dt
        axes[1, 0].plot(lag_times, autocorr, alpha=0.7, linewidth=1.5, 
                       label=f'η={eta}', color=colors[idx])
        
        # Fit damped oscillation to extract damping rate
        if eta > 0:  # Only fit for noisy cases
            fit_params = fit_damped_oscillation(lag_times, autocorr)
            if fit_params is not None:
                A, lam, omega, phi = fit_params
                damping_rates.append(lam)
                eta_nonzero.append(eta)
                # Plot fit on autocorrelation
                fit_curve = A * np.exp(-lam * lag_times) * np.cos(omega * lag_times + phi)
                axes[1, 0].plot(lag_times, fit_curve, '--', alpha=0.4, 
                              linewidth=1, color=colors[idx])
        
        # 4. Variance vs noise (computed across all data)
        variance = np.var(theta_steady)
        axes[1, 1].scatter(eta, variance, s=80, color=colors[idx], zorder=3)
    
    # Format subplot 1: Time series
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('θ (radians)')
    axes[0, 0].set_title('Sample Time Series')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Format subplot 2: PSD
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power Spectral Density')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].legend(fontsize=8, loc='best')
    axes[0, 1].grid(True, alpha=0.3, which='both')
    
    # Format subplot 3: Autocorrelation
    axes[1, 0].set_xlabel('Lag Time')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_title('Autocorrelation Function')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Format subplot 4: Variance vs noise
    axes[1, 1].plot(eta_values, [axes[1, 1].collections[i].get_offsets()[0, 1] 
                                  for i in range(len(axes[1, 1].collections))], 
                   'o-', linewidth=2, markersize=8, color='steelblue')
    axes[1, 1].set_xlabel('Noise Strength (η)')
    axes[1, 1].set_ylabel('Variance of θ')
    axes[1, 1].set_title('Variance vs Noise Strength')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    # Format subplot 5: Damping rate vs noise
    if len(damping_rates) > 0:
        axes[0, 2].loglog(eta_nonzero, damping_rates, 'o', markersize=8, color='darkred', zorder=3)
            
    axes[0, 2].set_xlabel('Noise Strength (η)')
    axes[0, 2].set_ylabel('Damping Rate (λ)')
    axes[0, 2].set_title('Autocorrelation Damping Rate')
    axes[0, 2].grid(True, alpha=0.3, which='both')
    
    # Format subplot 6: Correlation time vs noise
    if len(damping_rates) > 0:
        corr_times = [1/lam if lam > 0 else np.nan for lam in damping_rates]
        axes[1, 2].loglog(eta_nonzero, corr_times, 'o', markersize=8, color='darkorange', zorder=3)
                
    axes[1, 2].set_xlabel('Noise Strength (η)')
    axes[1, 2].set_ylabel('Correlation Time (τ_c)')
    axes[1, 2].set_title('Correlation Time')
    axes[1, 2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"noise_analysis_tau_{tau}_k_{k}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Analysis saved to: {output_file}")
    
    plt.close()

if __name__ == '__main__':
    # Parameters (should match the bash script)
    tau = 5
    k = 0.32
    theta0 = 1.5708
    dt = 0.1
    tmax = 10000
    
    analyze_timeseries(tau, k, theta0, dt, tmax)
