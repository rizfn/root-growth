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

// Deterministic DDE: dθ/dt = -k·sin(θ(t-τ))
// Designed for intermittency analysis near the chaos onset.
// Outputs dense θ(t) AFTER warmup so the full measurement window is on-attractor.

constexpr double DEFAULT_TAU       = 25.0;
constexpr double DEFAULT_K         = 0.19;
constexpr double DEFAULT_THETA0    = 1.5708;
constexpr double DEFAULT_DT        = 0.01;
constexpr double DEFAULT_RECORD_DT = 0.1;
constexpr double DEFAULT_T_WARMUP  = 5000.0;   // long warmup to settle on attractor
constexpr double DEFAULT_T_MEASURE = 50000.0;  // long measurement to accumulate laminar statistics

struct HistoryBuffer
{
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta) { times.push_back(t); values.push_back(theta); }

    double getDelayed(double t, double tau, double theta0) const
    {
        double target_time = t - tau;
        if (times.empty() || target_time <= times.front()) return theta0;
        if (target_time >= times.back()) return values.back();
        for (size_t i = 0; i < times.size() - 1; ++i)
        {
            if (times[i] <= target_time && target_time <= times[i + 1])
            {
                double alpha = (target_time - times[i]) / (times[i + 1] - times[i]);
                return values[i] + alpha * (values[i + 1] - values[i]);
            }
        }
        return values.back();
    }

    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0;
        while (!times.empty() && times.front() < cutoff)
        { times.pop_front(); values.pop_front(); }
    }
};

double heunStep(HistoryBuffer &history, double t, double theta,
                double dt, double tau, double theta0, double k)
{
    double theta_delayed = history.getDelayed(t, tau, theta0);
    double k1 = -k * std::sin(theta_delayed);
    double theta_pred = theta + k1 * dt;
    double t_next = t + dt;

    history.add(t_next, theta_pred);
    double k2 = -k * std::sin(history.getDelayed(t_next, tau, theta0));
    history.times.pop_back();
    history.values.pop_back();

    return theta + 0.5 * (k1 + k2) * dt;
}

void simulate(std::ofstream &file,
              double tau, double k, double theta0,
              double dt, double t_warmup, double t_measure, double record_dt)
{
    HistoryBuffer history;
    double t = 0.0, theta = theta0;
    history.add(t, theta);

    // ── Warmup (silent) ──────────────────────────────────────────
    int warmup_steps = static_cast<int>(t_warmup / dt);
    for (int step = 0; step < warmup_steps; ++step)
    {
        theta = heunStep(history, t, theta, dt, tau, theta0, k);
        t += dt;
        history.add(t, theta);
        if (step % 100 == 0) history.pruneOld(t, tau);
    }

    // ── Measurement phase ────────────────────────────────────────
    int record_steps  = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    int measure_steps = static_cast<int>(t_measure / dt);

    file << "time\ttheta\n";
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\n";

    for (int step = 0; step < measure_steps; ++step)
    {
        theta = heunStep(history, t, theta, dt, tau, theta0, k);
        t += dt;
        history.add(t, theta);

        if ((step + 1) % record_steps == 0)
            file << t << "\t" << theta << "\n";

        if (step % 100 == 0) history.pruneOld(t, tau);
    }
}

int main(int argc, char *argv[])
{
    double tau       = DEFAULT_TAU;
    double k         = DEFAULT_K;
    double theta0    = DEFAULT_THETA0;
    double dt        = DEFAULT_DT;
    double t_warmup  = DEFAULT_T_WARMUP;
    double t_measure = DEFAULT_T_MEASURE;
    double record_dt = DEFAULT_RECORD_DT;

    if (argc > 1) tau       = std::stod(argv[1]);
    if (argc > 2) k         = std::stod(argv[2]);
    if (argc > 3) theta0    = std::stod(argv[3]);
    if (argc > 4) dt        = std::stod(argv[4]);
    if (argc > 5) t_warmup  = std::stod(argv[5]);
    if (argc > 6) t_measure = std::stod(argv[6]);
    if (argc > 7) record_dt = std::stod(argv[7]);

    std::string exeDir = std::filesystem::path(argv[0]).parent_path().string();

    std::ostringstream ss;
    ss << exeDir << "/outputs/intermittency/k_sweep"
       << "/tau_" << tau << "_k_" << k << "_theta0_" << theta0
       << "_twarmup_" << t_warmup << "_tmeasure_" << t_measure
       << "_dt_" << record_dt << ".tsv";
    std::string filePath = ss.str();

    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file(filePath);
    simulate(file, tau, k, theta0, dt, t_warmup, t_measure, record_dt);
    file.close();

    return 0;
}
