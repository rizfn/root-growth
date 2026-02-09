#include <vector>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <random>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<double> normal_dist(0.0, 1.0);

// DDE parameters: dθ/dt = -k·sin(θ(t-τ)) + η·ξ(t)
constexpr double DEFAULT_TAU = 20.0;           // Time lag
constexpr double DEFAULT_K = 0.08;             // Gravitropic strength
constexpr double DEFAULT_ETA = 0.1;            // Noise strength
constexpr double DEFAULT_THETA0 = 1.5708;      // Initial angle (π/2 radians = 90 degrees)
constexpr double DEFAULT_DT = 0.1;             // Time step
constexpr double DEFAULT_T_MAX = 1000.0;       // Total simulation time

// History buffer to store past theta values
struct HistoryBuffer
{
    std::deque<double> times;
    std::deque<double> values;
    
    void add(double t, double theta)
    {
        times.push_back(t);
        values.push_back(theta);
    }
    
    // Get theta at time (t - tau) using linear interpolation
    double getDelayed(double t, double tau, double theta0) const
    {
        double target_time = t - tau;
        
        // If target time is before our history, return initial condition
        if (target_time <= 0.0 || times.empty())
            return theta0;
        
        // Find the two points to interpolate between
        for (size_t i = 0; i < times.size() - 1; ++i)
        {
            if (times[i] <= target_time && target_time <= times[i + 1])
            {
                // Linear interpolation
                double t1 = times[i];
                double t2 = times[i + 1];
                double v1 = values[i];
                double v2 = values[i + 1];
                double alpha = (target_time - t1) / (t2 - t1);
                return v1 + alpha * (v2 - v1);
            }
        }
        
        // If target time is after our last point, return last value
        return values.back();
    }
    
    // Remove old history to save memory (keep only what's needed for delay)
    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0; // Keep a bit extra for safety
        while (!times.empty() && times.front() < cutoff)
        {
            times.pop_front();
            values.pop_front();
        }
    }
};

// Solve DDE using Heun's method (stochastic RK2 - strong order 1.0)
// For additive noise SDDEs: RK2 for deterministic part, optimal noise treatment
void solveDDE(std::ofstream &file, double tau, double k, double eta, double theta0, double dt, double t_max)
{
    HistoryBuffer history;
    
    double t = 0.0;
    double theta = theta0;
    
    // Write header
    file << "time\ttheta\n";
    
    // Write initial condition
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\n";
    history.add(t, theta);
    
    int n_steps = static_cast<int>(t_max / dt);
    
    for (int step = 0; step < n_steps; ++step)
    {
        // Heun's method (stochastic RK2) for SDDE with additive noise
        // Predictor-corrector approach for deterministic part
        
        // Generate noise term (same for predictor and corrector)
        double noise = eta * normal_dist(gen) * std::sqrt(dt);
        
        // Predictor step: evaluate derivative at current delayed state
        double theta_delayed = history.getDelayed(t, tau, theta0);
        double k1 = -k * std::sin(theta_delayed);
        
        // Predicted value at t+dt
        double theta_pred = theta + k1 * dt + noise;
        double t_next = t + dt;
        
        // Temporarily add predictor to history for delay interpolation
        history.add(t_next, theta_pred);
        
        // Corrector step: evaluate derivative at predicted delayed state
        double theta_delayed_pred = history.getDelayed(t_next, tau, theta0);
        double k2 = -k * std::sin(theta_delayed_pred);
        
        // Remove temporary predictor from history
        history.times.pop_back();
        history.values.pop_back();
        
        // Heun's corrector: average of slopes
        double deterministic = 0.5 * (k1 + k2);
        
        // Update theta with corrected deterministic term and noise
        theta = theta + deterministic * dt + noise;
        t = t_next;
        
        // Store corrected value in history
        history.add(t, theta);
        
        // Write to file
        file << t << "\t" << theta << "\n";
        
        // Prune old history periodically
        if (step % 100 == 0)
        {
            history.pruneOld(t, tau);
        }
        
        // Print progress every 10%
        if (step % (n_steps / 10) == 0)
        {
            double progress = 100.0 * step / n_steps;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
        }
    }
    
    std::cout << "\rProgress: 100.0%  " << std::endl;
}

int main(int argc, char *argv[])
{
    double tau = DEFAULT_TAU;
    double k = DEFAULT_K;
    double eta = DEFAULT_ETA;
    double theta0 = DEFAULT_THETA0;
    double dt = DEFAULT_DT;
    double t_max = DEFAULT_T_MAX;
    int sim_no = 0;

    if (argc > 1)
        tau = std::stod(argv[1]);
    if (argc > 2)
        k = std::stod(argv[2]);
    if (argc > 3)
        eta = std::stod(argv[3]);
    if (argc > 4)
        theta0 = std::stod(argv[4]);
    if (argc > 5)
        dt = std::stod(argv[5]);
    if (argc > 6)
        t_max = std::stod(argv[6]);
    if (argc > 7)
        sim_no = std::stoi(argv[7]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    
    // Create folder for this parameter set
    std::ostringstream folderStream;
    folderStream << exeDir << "/outputs/SDDETimeseries/long/tau_" << tau
                 << "_k_" << k << "_theta0_" << theta0
                 << "_dt_" << dt << "_tmax_" << t_max;
    std::string folderPath = folderStream.str();
    
    // Create file path within folder
    std::ostringstream filePathStream;
    filePathStream << folderPath << "/eta_" << eta << "_simNo_" << sim_no << ".tsv";
    std::string filePath = filePathStream.str();

    // Create directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);

    solveDDE(file, tau, k, eta, theta0, dt, t_max);

    file.close();

    std::cout << "SDDE solution complete. Results saved to: " << filePath << std::endl;

    return 0;
}