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
constexpr double DEFAULT_K = 5.2;
constexpr double DEFAULT_ETA = 0;
constexpr double DEFAULT_THETA0 = 1.5708;
constexpr double DEFAULT_DT = 0.01;
constexpr double DEFAULT_T_WARMUP = 500.0;
constexpr double DEFAULT_T_LYAP = 200.0;
constexpr double DEFAULT_DELTA0 = 1e-8;
constexpr unsigned int DEFAULT_SEED = 42;
constexpr int DEFAULT_SAVE_TS = 1;

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

double rk4DriftStep(const HistoryBuffer &history, double t, double dt, double tau, double theta0, double k)
{
    const double d1 = -k * std::sin(history.getDelayed(t, tau, theta0));
    const double d2 = -k * std::sin(history.getDelayed(t + 0.5 * dt, tau, theta0));
    const double d3 = -k * std::sin(history.getDelayed(t + 0.5 * dt, tau, theta0));
    const double d4 = -k * std::sin(history.getDelayed(t + dt, tau, theta0));
    return (dt / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4);
}

struct DeviationPoint
{
    double time = 0.0;
    double separation = std::numeric_limits<double>::quiet_NaN();
};

struct DeviationOutput
{
    double lambda = std::numeric_limits<double>::quiet_NaN();
    std::vector<DeviationPoint> deviation_trace;
};

DeviationOutput runDeviationTrace(double tau,
                                  double k,
                                  double eta,
                                  double theta0,
                                  double dt,
                                  double t_warmup,
                                  double t_lyap,
                                  double delta0,
                                  unsigned int seed,
                                  bool save_timeseries)
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

    DeviationOutput out;
    if (save_timeseries)
        out.deviation_trace.reserve(static_cast<size_t>(measure_steps + 2));

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

        const double sep = std::abs(theta_pert - theta_base);

        if (save_timeseries && std::isfinite(sep) && sep > 0.0)
        {
            DeviationPoint p;
            p.time = t;
            p.separation = sep;
            out.deviation_trace.push_back(p);
        }

        if (step % 100 == 0)
        {
            base_history.pruneOld(t, tau);
            pert_history.pruneOld(t, tau);
        }
    }

    return out;
}

double estimateLyapunovFromTrace(const std::vector<DeviationPoint>& trace, double tau, double dt)
{
    if (trace.size() < 10)
        return std::numeric_limits<double>::quiet_NaN();

    // Extract separation values
    std::vector<double> sep;
    for (const auto& p : trace)
    {
        if (std::isfinite(p.separation) && p.separation > 0.0)
            sep.push_back(p.separation);
    }

    if (sep.size() < 10)
        return std::numeric_limits<double>::quiet_NaN();

    // Forward smooth over 4*tau
    const int window = std::max(1, static_cast<int>(std::round((4.0 * tau) / dt)));
    if (static_cast<int>(sep.size()) < window + 5)
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> avg_sep;
    for (int i = 0; i + window <= static_cast<int>(sep.size()); ++i)
    {
        double sum = 0.0;
        for (int j = i; j < i + window; ++j)
            sum += sep[j];
        avg_sep.push_back(sum / window);
    }

    if (avg_sep.size() < 10)
        return std::numeric_limits<double>::quiet_NaN();

    // Fit slope to first 30% of smoothed trace in log-space
    const int fit_count = std::max(6, static_cast<int>(0.3 * avg_sep.size()));
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    int valid_count = 0;

    for (int i = 0; i < fit_count; ++i)
    {
        if (avg_sep[i] <= 0.0)
            continue;
        double x = static_cast<double>(i) * dt;
        double y = std::log(avg_sep[i]);
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        ++valid_count;
    }

    if (valid_count < 6)
        return std::numeric_limits<double>::quiet_NaN();

    double slope = (valid_count * sum_xy - sum_x * sum_y) / (valid_count * sum_x2 - sum_x * sum_x);
    return slope;
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
    double delta0 = DEFAULT_DELTA0;
    unsigned int seed = DEFAULT_SEED;
    int save_timeseries = DEFAULT_SAVE_TS;

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
        t_warmup = std::stod(argv[6]);
    if (argc > 7)
        t_lyap = std::stod(argv[7]);
    if (argc > 11)
    {
        // Legacy signature keeps renorm_dt at argv[8] (ignored here).
        delta0 = std::stod(argv[9]);
        seed = static_cast<unsigned int>(std::stoul(argv[10]));
        save_timeseries = std::stoi(argv[11]);
    }
    else
    {
        // Simplified signature: ... t_lyap delta0 seed save_timeseries
        if (argc > 8)
            delta0 = std::stod(argv[8]);
        if (argc > 9)
            seed = static_cast<unsigned int>(std::stoul(argv[9]));
        if (argc > 10)
            save_timeseries = std::stoi(argv[10]);
    }

    std::string exe_path = argv[0];
    std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();

    const std::string base_out = exe_dir + "/outputs/lyapunov_test";
    const std::string summary_dir = base_out + "/run_summaries";
    const std::string trace_dir = base_out + "/deviation_timeseries";

    std::filesystem::create_directories(summary_dir);
    if (save_timeseries != 0)
        std::filesystem::create_directories(trace_dir);

    const DeviationOutput out = runDeviationTrace(
        tau,
        k,
        eta,
        theta0,
        dt,
        t_warmup,
        t_lyap,
        delta0,
        seed,
        save_timeseries != 0);

    // Estimate lambda from the trace
    double lambda = estimateLyapunovFromTrace(out.deviation_trace, tau, dt);

    const std::string slug = paramSlug(tau, k, eta, seed);

    {
        std::ofstream f(summary_dir + "/" + slug + ".tsv");
        f << "tau\tk\teta\ttheta0\tdt\tt_warmup\tt_lyap\tdelta0\tseed\tavg_period\tlambda\n";
        f << std::setprecision(10) << tau << "\t" << k << "\t" << eta << "\t" << theta0 << "\t" << dt << "\t"
          << t_warmup << "\t" << t_lyap << "\t" << delta0 << "\t" << seed << "\t" << (4.0 * tau) << "\t" << lambda << "\n";
    }

    if (save_timeseries != 0)
    {
        std::ofstream f(trace_dir + "/" + slug + ".tsv");
                f << "time\tseparation\n";
        for (const auto &p : out.deviation_trace)
        {
                        f << std::setprecision(10) << p.time << "\t" << p.separation << "\n";
        }
    }

    return 0;
}
