#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    return 0;
}();

constexpr double DEFAULT_TAU = 1.0;
constexpr double DEFAULT_K = 4.5;
constexpr double DEFAULT_ETA = 1e-4;
constexpr double DEFAULT_THETA0 = 1.5708;
constexpr double DEFAULT_DT = 0.01;
constexpr double DEFAULT_T_WARMUP = 500.0;
constexpr double DEFAULT_T_LYAP = 10.0;   // 1000 steps at dt=0.01
constexpr double DEFAULT_RENORM_DT = 1.0;
constexpr double DEFAULT_DELTA0 = 1e-8;
constexpr unsigned int DEFAULT_SEED = 42;
constexpr int DEFAULT_SAVE_TRACE = 0;

struct HistoryBuffer
{
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta)
    {
        times.push_back(t);
        values.push_back(theta);
    }

    double getDelayed(double t, double tau, double theta0) const
    {
        const double target_time = t - tau;

        if (times.empty() || target_time <= times.front())
            return theta0;

        if (target_time >= times.back())
            return values.back();

        for (size_t i = 0; i + 1 < times.size(); ++i)
        {
            if (times[i] <= target_time && target_time <= times[i + 1])
            {
                const double t1 = times[i];
                const double t2 = times[i + 1];
                const double v1 = values[i];
                const double v2 = values[i + 1];
                const double alpha = (target_time - t1) / (t2 - t1);
                return v1 + alpha * (v2 - v1);
            }
        }

        return values.back();
    }

    void pruneOld(double current_time, double tau)
    {
        const double cutoff = current_time - tau - 2.0;
        while (!times.empty() && times.front() < cutoff)
        {
            times.pop_front();
            values.pop_front();
        }
    }
};

// Deterministic drift uses delayed state only. With additive noise,
// RK4 drift + exact Wiener increment gives high weak-order accuracy.
double rk4DriftStep(const HistoryBuffer &history, double t, double dt, double tau, double theta0, double k)
{
    const double d1 = -k * std::sin(history.getDelayed(t, tau, theta0));
    const double d2 = -k * std::sin(history.getDelayed(t + 0.5 * dt, tau, theta0));
    const double d3 = -k * std::sin(history.getDelayed(t + 0.5 * dt, tau, theta0));
    const double d4 = -k * std::sin(history.getDelayed(t + dt, tau, theta0));
    return (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4);
}

double historyDistanceRms(const HistoryBuffer &baseline, const HistoryBuffer &perturbed)
{
    const size_t n = std::min(baseline.values.size(), perturbed.values.size());
    if (n == 0)
        return 0.0;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double dv = perturbed.values[i] - baseline.values[i];
        sum_sq += dv * dv;
    }
    return std::sqrt(sum_sq / static_cast<double>(n));
}

void renormalizePerturbation(const HistoryBuffer &baseline, HistoryBuffer &perturbed, double target_delta)
{
    const size_t n = std::min(baseline.values.size(), perturbed.values.size());
    if (n == 0)
        return;

    const double dist = historyDistanceRms(baseline, perturbed);
    if (!std::isfinite(dist) || dist <= 0.0)
        return;

    const double scale = target_delta / dist;
    for (size_t i = 0; i < n; ++i)
    {
        const double diff = perturbed.values[i] - baseline.values[i];
        perturbed.values[i] = baseline.values[i] + diff * scale;
    }
}

struct LyapunovOutput
{
    double lambda = std::numeric_limits<double>::quiet_NaN();
    int n_renorm = 0;
    std::vector<std::pair<double, double>> trace;
};

LyapunovOutput runLyapunov(double tau,
                           double k,
                           double eta,
                           double theta0,
                           double dt,
                           double t_warmup,
                           double t_lyap,
                           double renorm_dt,
                           double delta0,
                           unsigned int seed,
                           bool save_trace)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    HistoryBuffer base_history;
    HistoryBuffer pert_history;

    double t = 0.0;
    double theta_base = theta0;
    base_history.add(t, theta_base);

    const int warmup_steps = static_cast<int>(std::round(t_warmup / dt));
    for (int step = 0; step < warmup_steps; ++step)
    {
        const double drift = rk4DriftStep(base_history, t, dt, tau, theta0, k);
        const double dW = std::sqrt(dt) * normal(rng);
        theta_base += drift + eta * dW;
        t += dt;
        base_history.add(t, theta_base);

        if (step % 100 == 0)
            base_history.pruneOld(t, tau);
    }

    pert_history = base_history;
    for (double &v : pert_history.values)
        v += delta0;
    double theta_pert = theta_base + delta0;

    const int measure_steps = static_cast<int>(std::round(t_lyap / dt));
    const int renorm_steps = std::max(1, static_cast<int>(std::round(renorm_dt / dt)));

    LyapunovOutput out;
    if (save_trace)
        out.trace.reserve(static_cast<size_t>(measure_steps / renorm_steps + 2));

    double log_sum = 0.0;

    for (int step = 0; step < measure_steps; ++step)
    {
        const double dW = std::sqrt(dt) * normal(rng);

        const double drift_base = rk4DriftStep(base_history, t, dt, tau, theta0, k);
        const double drift_pert = rk4DriftStep(pert_history, t, dt, tau, theta0, k);

        theta_base += drift_base + eta * dW;
        theta_pert += drift_pert + eta * dW;
        t += dt;

        base_history.add(t, theta_base);
        pert_history.add(t, theta_pert);

        if ((step + 1) % renorm_steps == 0)
        {
            const double dist = historyDistanceRms(base_history, pert_history);
            if (std::isfinite(dist) && dist > 0.0)
            {
                log_sum += std::log(dist / delta0);
                ++out.n_renorm;
                if (save_trace)
                    out.trace.push_back({t, std::log(dist)});
                renormalizePerturbation(base_history, pert_history, delta0);
                theta_pert = base_history.values.back() + (pert_history.values.back() - base_history.values.back());
            }
        }

        if (step % 100 == 0)
        {
            base_history.pruneOld(t, tau);
            pert_history.pruneOld(t, tau);
        }
    }

    if (out.n_renorm > 0)
        out.lambda = log_sum / (static_cast<double>(out.n_renorm) * renorm_dt);

    return out;
}

std::string paramSlug(double tau, double k, double eta, unsigned int seed)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "tau_" << tau << "_k_" << k << "_eta_" << eta << "_seed_" << seed;
    return oss.str();
}

int main(int argc, char *argv[])
{
    double tau = DEFAULT_TAU;
    double k = DEFAULT_K;
    double eta = DEFAULT_ETA;
    double theta0 = DEFAULT_THETA0;
    double dt = DEFAULT_DT;
    double t_warmup = DEFAULT_T_WARMUP;
    double t_lyap = DEFAULT_T_LYAP;
    double renorm_dt = DEFAULT_RENORM_DT;
    double delta0 = DEFAULT_DELTA0;
    unsigned int seed = DEFAULT_SEED;
    int save_trace = DEFAULT_SAVE_TRACE;

    if (argc > 1) tau = std::stod(argv[1]);
    if (argc > 2) k = std::stod(argv[2]);
    if (argc > 3) eta = std::stod(argv[3]);
    if (argc > 4) theta0 = std::stod(argv[4]);
    if (argc > 5) dt = std::stod(argv[5]);
    if (argc > 6) t_warmup = std::stod(argv[6]);
    if (argc > 7) t_lyap = std::stod(argv[7]);
    if (argc > 8) renorm_dt = std::stod(argv[8]);
    if (argc > 9) delta0 = std::stod(argv[9]);
    if (argc > 10) seed = static_cast<unsigned int>(std::stoul(argv[10]));
    if (argc > 11) save_trace = std::stoi(argv[11]);

    std::string exe_path = argv[0];
    std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();

    const std::string base_out = exe_dir + "/outputs/lyapunov_scan";
    const std::string summary_dir = base_out + "/run_summaries";
    const std::string trace_dir = base_out + "/traces";

    std::filesystem::create_directories(summary_dir);
    if (save_trace != 0)
        std::filesystem::create_directories(trace_dir);

    const LyapunovOutput out = runLyapunov(
        tau,
        k,
        eta,
        theta0,
        dt,
        t_warmup,
        t_lyap,
        renorm_dt,
        delta0,
        seed,
        save_trace != 0);

    const std::string slug = paramSlug(tau, k, eta, seed);

    {
        std::ofstream f(summary_dir + "/" + slug + ".tsv");
        f << "tau\tk\teta\ttheta0\tdt\tt_warmup\tt_lyap\trenorm_dt\tdelta0\tseed"
             "\tlambda\tn_renorm\n";
        f << std::setprecision(10)
          << tau << "\t" << k << "\t" << eta << "\t" << theta0 << "\t" << dt << "\t"
          << t_warmup << "\t" << t_lyap << "\t" << renorm_dt << "\t" << delta0 << "\t" << seed << "\t"
          << out.lambda << "\t" << out.n_renorm << "\n";
    }

    if (save_trace != 0)
    {
        std::ofstream f(trace_dir + "/" + slug + ".tsv");
        f << "time\tlog_distance\n";
        for (const auto &p : out.trace)
            f << std::setprecision(10) << p.first << "\t" << p.second << "\n";
    }

    return 0;
}
