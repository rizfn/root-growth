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
// Initial condition: θ(t) = A·sin(2π t / T_ic)  for t ∈ [-τ, 0]
constexpr double DEFAULT_TAU = 25.0;
constexpr double DEFAULT_K = 0.2;
constexpr double DEFAULT_ETA = 0.0;
constexpr double DEFAULT_DT = 0.01;
constexpr double DEFAULT_RECORD_DT = 0.1;
constexpr double DEFAULT_T_MAX = 4000.0;

constexpr double DEFAULT_IC_AMPLITUDE = 1.0;   // A
constexpr double DEFAULT_IC_PERIOD = 10.0;     // T_ic

constexpr double PI = 3.14159265358979323846;

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
    
    // Get theta at time target_time using linear interpolation.
    // For times before the start of history, fall back to the IC function.
    double getDelayed(double t, double tau, double ic_amp, double ic_period) const
    {
        double target_time = t - tau;
        
        // If the buffer has data covering target_time, interpolate
        if (!times.empty() && target_time >= times.front())
        {
            // Binary search for efficiency (history is sorted)
            auto it = std::lower_bound(times.begin(), times.end(), target_time);
            if (it == times.end())
                return values.back();
            if (it == times.begin())
                return values.front();

            size_t idx = std::distance(times.begin(), it);
            double t2 = times[idx];
            double t1 = times[idx - 1];
            double v2 = values[idx];
            double v1 = values[idx - 1];
            double alpha = (target_time - t1) / (t2 - t1);
            return v1 + alpha * (v2 - v1);
        }
        
        // Fall back to sinusoidal IC for t < -tau .. 0
        return ic_amp * std::sin(2.0 * PI * target_time / ic_period);
    }
    
    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0;
        while (!times.empty() && times.front() < cutoff)
        {
            times.pop_front();
            values.pop_front();
        }
    }
};

// Solve DDE using Heun's method with sinusoidal initial condition
void solveDDE(std::ofstream &file, double tau, double k, double eta,
              double ic_amp, double ic_period,
              double dt, double t_max, double record_dt)
{
    HistoryBuffer history;

    int record_steps = std::max(1, static_cast<int>(std::round(record_dt / dt)));

    // ── Write header ──
    file << "time\ttheta\n";

    // ── Write the initial condition on [-tau, 0] at record_dt intervals ──
    {
        int ic_steps = static_cast<int>(std::round(tau / record_dt));
        for (int i = ic_steps; i >= 0; --i)
        {
            double t_ic = -i * record_dt;
            double theta_ic = ic_amp * std::sin(2.0 * PI * t_ic / ic_period);
            file << std::fixed << std::setprecision(6) << t_ic << "\t" << theta_ic << "\n";
        }
    }

    // ── Pre-fill the history buffer on [-tau, 0] at dt resolution ──
    {
        int n_ic = static_cast<int>(std::round(tau / dt));
        for (int i = n_ic; i >= 0; --i)
        {
            double t_ic = -i * dt;
            double theta_ic = ic_amp * std::sin(2.0 * PI * t_ic / ic_period);
            history.add(t_ic, theta_ic);
        }
    }

    double t = 0.0;
    double theta = ic_amp * std::sin(0.0); // = 0 at t=0

    int n_steps = static_cast<int>(t_max / dt);

    for (int step = 0; step < n_steps; ++step)
    {
        double noise = eta * normal_dist(gen) * std::sqrt(dt);

        // Predictor
        double theta_delayed = history.getDelayed(t, tau, ic_amp, ic_period);
        double k1 = -k * std::sin(theta_delayed);
        double theta_pred = theta + k1 * dt + noise;
        double t_next = t + dt;

        history.add(t_next, theta_pred);

        // Corrector
        double theta_delayed_pred = history.getDelayed(t_next, tau, ic_amp, ic_period);
        double k2 = -k * std::sin(theta_delayed_pred);

        history.times.pop_back();
        history.values.pop_back();

        double deterministic = 0.5 * (k1 + k2);
        theta = theta + deterministic * dt + noise;
        t = t_next;

        history.add(t, theta);

        if ((step + 1) % record_steps == 0)
        {
            file << t << "\t" << theta << "\n";
        }

        if (step % 100 == 0)
        {
            history.pruneOld(t, tau);
        }

        if (step % (n_steps / 10) == 0)
        {
            double progress = 100.0 * step / n_steps;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                      << progress << "%" << std::flush;
        }
    }
    std::cout << "\rProgress: 100.0%  " << std::endl;
}

int main(int argc, char *argv[])
{
    double tau        = DEFAULT_TAU;
    double k          = DEFAULT_K;
    double eta        = DEFAULT_ETA;
    double ic_amp     = DEFAULT_IC_AMPLITUDE;
    double ic_period  = DEFAULT_IC_PERIOD;
    double dt         = DEFAULT_DT;
    double t_max      = DEFAULT_T_MAX;
    int    sim_no     = 0;
    double record_dt  = DEFAULT_RECORD_DT;

    // CLI: tau k eta ic_amp ic_period dt t_max sim_no record_dt
    if (argc > 1) tau       = std::stod(argv[1]);
    if (argc > 2) k         = std::stod(argv[2]);
    if (argc > 3) eta       = std::stod(argv[3]);
    if (argc > 4) ic_amp    = std::stod(argv[4]);
    if (argc > 5) ic_period = std::stod(argv[5]);
    if (argc > 6) dt        = std::stod(argv[6]);
    if (argc > 7) t_max     = std::stod(argv[7]);
    if (argc > 8) sim_no    = std::stoi(argv[8]);
    if (argc > 9) record_dt = std::stod(argv[9]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();

    // Output folder structure:
    //   outputs/SDDETimeseries/periodic_ic/tau_25_k_0.2_dt_0.1_tmax_4000/
    //     amp_1_period_10_eta_0_simNo_0.tsv
    std::ostringstream folderStream;
    folderStream << exeDir << "/outputs/SDDETimeseries/periodic_ic/tau_" << tau
                 << "_k_" << k << "_dt_" << record_dt << "_tmax_" << t_max;
    std::string folderPath = folderStream.str();

    std::ostringstream filePathStream;
    filePathStream << folderPath << "/amp_" << ic_amp
                   << "_period_" << ic_period
                   << "_eta_" << eta
                   << "_simNo_" << sim_no << ".tsv";
    std::string filePath = filePathStream.str();

    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);

    std::cout << "tau=" << tau << " k=" << k << " eta=" << eta
              << " A=" << ic_amp << " T_ic=" << ic_period << std::endl;

    solveDDE(file, tau, k, eta, ic_amp, ic_period, dt, t_max, record_dt);

    file.close();
    std::cout << "Done. Saved: " << filePath << std::endl;

    return 0;
}
