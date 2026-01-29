#include <vector>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

// DDE parameters: dθ/dt = -k·sin(θ(t-τ))
constexpr double DEFAULT_TAU = 20.0;           // Time lag
constexpr double DEFAULT_K = 0.08;             // Gravitropic strength
constexpr double DEFAULT_THETA0 = 1.5708;      // Initial angle (π/2 radians = 90 degrees)
constexpr double DEFAULT_DT = 0.1;             // Time step
constexpr double DEFAULT_T_MAX = 1000.0;       // Total simulation time

// History buffer to store past theta values
struct HistoryBuffer
{
    std::vector<double> times;
    std::vector<double> values;
    size_t start_idx = 0;  // Track valid data start
    
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
        if (target_time <= 0.0 || start_idx >= times.size())
            return theta0;
        
        // Find the two points to interpolate between
        for (size_t i = start_idx; i < times.size() - 1; ++i)
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
    
    // Mark old history as invalid without actually removing (O(1) operation)
    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0;
        while (start_idx < times.size() && times[start_idx] < cutoff)
        {
            ++start_idx;
        }
        
        // Occasionally compact the vectors to free memory
        if (start_idx > 5000)
        {
            times.erase(times.begin(), times.begin() + start_idx);
            values.erase(values.begin(), values.begin() + start_idx);
            start_idx = 0;
        }
    }
};

// DDE right-hand side: dθ/dt = -k·sin(θ(t-τ))
inline double dde_rhs(double theta_delayed, double k)
{
    return -k * std::sin(theta_delayed);
}

// Solve DDE using RK4 method
void solveDDE(std::ofstream &file, double tau, double k, double theta0, double dt, double t_max)
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
        // RK4 method for DDE
        double theta_delayed_k1 = history.getDelayed(t, tau, theta0);
        double k1 = dde_rhs(theta_delayed_k1, k);
        
        double theta_delayed_k2 = history.getDelayed(t + 0.5 * dt, tau, theta0);
        double k2 = dde_rhs(theta_delayed_k2, k);
        
        double theta_delayed_k3 = history.getDelayed(t + 0.5 * dt, tau, theta0);
        double k3 = dde_rhs(theta_delayed_k3, k);
        
        double theta_delayed_k4 = history.getDelayed(t + dt, tau, theta0);
        double k4 = dde_rhs(theta_delayed_k4, k);
        
        // Update theta
        theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t = t + dt;
        
        // Store in history
        history.add(t, theta);
        
        // Write to file
        file << t << "\t" << theta << "\n";
        
        // Prune old history less frequently for better performance
        if (step % 1000 == 0)
        {
            history.pruneOld(t, tau);
            
            // Print progress
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
    double theta0 = DEFAULT_THETA0;
    double dt = DEFAULT_DT;
    double t_max = DEFAULT_T_MAX;

    if (argc > 1)
        tau = std::stod(argv[1]);
    if (argc > 2)
        k = std::stod(argv[2]);
    if (argc > 3)
        theta0 = std::stod(argv[3]);
    if (argc > 4)
        dt = std::stod(argv[4]);
    if (argc > 5)
        t_max = std::stod(argv[5]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/delayDETimeseries/tau_" << tau
                   << "_k_" << k << "_theta0_" << theta0
                   << "_dt_" << dt << "_tmax_" << t_max << ".tsv";
    std::string filePath = filePathStream.str();

    // Create directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);

    solveDDE(file, tau, k, theta0, dt, t_max);

    file.close();

    std::cout << "DDE solution complete. Results saved to: " << filePath << std::endl;

    return 0;
}