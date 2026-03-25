#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
constexpr double DEFAULT_T_WARMUP = 1000.0;
constexpr double DEFAULT_T_LAMINAR = 20000.0;
constexpr double DEFAULT_RECORD_DT = 0.1;
constexpr double PI = 3.14159265358979323846;
constexpr double DEFAULT_LAMINAR_THRESHOLD = PI / 2.0;
constexpr double DEFAULT_MIN_PERIODS = 1.0;
constexpr unsigned int DEFAULT_SEED = 42;
constexpr int DEFAULT_SAVE_TIMESERIES = 0;

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

std::vector<double> movingAverage(const std::vector<double> &x, int window)
{
    if (x.empty())
        return {};

    window = std::max(1, window);
    const int n = static_cast<int>(x.size());

    std::vector<double> y(n, 0.0);
    std::vector<double> prefix(n + 1, 0.0);
    for (int i = 0; i < n; ++i)
        prefix[i + 1] = prefix[i] + x[i];

    for (int i = 0; i < n; ++i)
    {
        const int a = std::max(0, i - window / 2);
        const int b = std::min(n, i + (window + 1) / 2);
        const double denom = static_cast<double>(b - a);
        y[i] = (prefix[b] - prefix[a]) / denom;
    }

    return y;
}

std::vector<double> detectLaminarLengths(const std::vector<double> &theta,
                                         double tau,
                                         double record_dt,
                                         double jump_threshold,
                                         double min_periods)
{
    if (theta.size() < 4)
        return {};

    const int period_samples = std::max(1, static_cast<int>(std::round((4.0 * tau) / record_dt)));
    const int min_samples = std::max(1, static_cast<int>(std::round(min_periods * period_samples)));

    const std::vector<double> center = movingAverage(theta, period_samples);
    if (center.size() <= static_cast<size_t>(period_samples))
        return {};

    std::vector<char> is_laminar(center.size() - period_samples, 0);
    for (size_t i = 0; i < is_laminar.size(); ++i)
    {
        const double dc = std::fabs(center[i + period_samples] - center[i]);
        is_laminar[i] = (dc < jump_threshold) ? 1 : 0;
    }

    std::vector<double> lengths;
    size_t i = 0;
    while (i < is_laminar.size())
    {
        if (!is_laminar[i])
        {
            ++i;
            continue;
        }

        const size_t start = i;
        while (i < is_laminar.size() && is_laminar[i])
            ++i;

        const int run_len = static_cast<int>(i - start);
        if (run_len >= min_samples)
            lengths.push_back(run_len * record_dt);
    }

    return lengths;
}

std::pair<double, double> fitPowerLawAlpha(const std::vector<double> &lengths, double ell_min)
{
    std::vector<double> x;
    x.reserve(lengths.size());
    for (double v : lengths)
    {
        if (v >= ell_min)
            x.push_back(v);
    }

    if (x.size() < 10)
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    double denom = 0.0;
    for (double v : x)
        denom += std::log(v / (ell_min - 0.5));

    if (denom <= 0.0)
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    const double alpha = 1.0 + static_cast<double>(x.size()) / denom;
    const double se = (alpha - 1.0) / std::sqrt(static_cast<double>(x.size()));
    return {alpha, se};
}

struct LaminarOutput
{
    int n_laminar = 0;
    double laminar_mean = std::numeric_limits<double>::quiet_NaN();
    double laminar_median = std::numeric_limits<double>::quiet_NaN();
    double tail_alpha = std::numeric_limits<double>::quiet_NaN();
    double tail_alpha_se = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> laminar_lengths;
};

LaminarOutput runLaminar(double tau,
                         double k,
                         double eta,
                         double theta0,
                         double dt,
                         double t_warmup,
                         double t_laminar,
                         double record_dt,
                         double laminar_threshold,
                         double min_periods,
                         unsigned int seed,
                         std::vector<std::pair<double, double>> *timeseries_out)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    HistoryBuffer history;
    double t = 0.0;
    double theta = theta0;
    history.add(t, theta);

    const int warmup_steps = static_cast<int>(std::round(t_warmup / dt));
    for (int step = 0; step < warmup_steps; ++step)
    {
        const double drift = rk4DriftStep(history, t, dt, tau, theta0, k);
        const double dW = std::sqrt(dt) * normal(rng);
        theta += drift + eta * dW;
        t += dt;
        history.add(t, theta);

        if (step % 100 == 0)
            history.pruneOld(t, tau);
    }

    const int measure_steps = static_cast<int>(std::round(t_laminar / dt));
    const int record_steps = std::max(1, static_cast<int>(std::round(record_dt / dt)));

    std::vector<double> theta_records;
    theta_records.reserve(static_cast<size_t>(measure_steps / record_steps + 2));

    if (timeseries_out != nullptr)
    {
        timeseries_out->clear();
        timeseries_out->reserve(static_cast<size_t>(measure_steps / record_steps + 2));
        timeseries_out->push_back({t, theta});
    }
    theta_records.push_back(theta);

    for (int step = 0; step < measure_steps; ++step)
    {
        const double drift = rk4DriftStep(history, t, dt, tau, theta0, k);
        const double dW = std::sqrt(dt) * normal(rng);
        theta += drift + eta * dW;
        t += dt;

        history.add(t, theta);

        if ((step + 1) % record_steps == 0)
        {
            theta_records.push_back(theta);
            if (timeseries_out != nullptr)
                timeseries_out->push_back({t, theta});
        }

        if (step % 100 == 0)
            history.pruneOld(t, tau);
    }

    LaminarOutput out;
    out.laminar_lengths = detectLaminarLengths(theta_records, tau, record_dt, laminar_threshold, min_periods);
    out.n_laminar = static_cast<int>(out.laminar_lengths.size());

    if (!out.laminar_lengths.empty())
    {
        const double sum = std::accumulate(out.laminar_lengths.begin(), out.laminar_lengths.end(), 0.0);
        out.laminar_mean = sum / static_cast<double>(out.laminar_lengths.size());

        std::vector<double> sorted = out.laminar_lengths;
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        if (n % 2 == 1)
            out.laminar_median = sorted[n / 2];
        else
            out.laminar_median = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);

        const double ell_min = std::max(2.0 * tau, 10.0 * record_dt);
        const auto [alpha, se] = fitPowerLawAlpha(out.laminar_lengths, ell_min);
        out.tail_alpha = alpha;
        out.tail_alpha_se = se;
    }

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
    double t_laminar = DEFAULT_T_LAMINAR;
    double record_dt = DEFAULT_RECORD_DT;
    double laminar_threshold = DEFAULT_LAMINAR_THRESHOLD;
    double min_periods = DEFAULT_MIN_PERIODS;
    unsigned int seed = DEFAULT_SEED;
    int save_timeseries = DEFAULT_SAVE_TIMESERIES;

    if (argc > 1) tau = std::stod(argv[1]);
    if (argc > 2) k = std::stod(argv[2]);
    if (argc > 3) eta = std::stod(argv[3]);
    if (argc > 4) theta0 = std::stod(argv[4]);
    if (argc > 5) dt = std::stod(argv[5]);
    if (argc > 6) t_warmup = std::stod(argv[6]);
    if (argc > 7) t_laminar = std::stod(argv[7]);
    if (argc > 8) record_dt = std::stod(argv[8]);
    if (argc > 9) laminar_threshold = std::stod(argv[9]);
    if (argc > 10) min_periods = std::stod(argv[10]);
    if (argc > 11) seed = static_cast<unsigned int>(std::stoul(argv[11]));
    if (argc > 12) save_timeseries = std::stoi(argv[12]);

    std::string exe_path = argv[0];
    std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();

    const std::string base_out = exe_dir + "/outputs/laminar_scan";
    const std::string summary_dir = base_out + "/run_summaries";
    const std::string laminar_dir = base_out + "/laminar_lengths";
    const std::string ts_dir = base_out + "/timeseries";

    std::filesystem::create_directories(summary_dir);
    std::filesystem::create_directories(laminar_dir);
    if (save_timeseries != 0)
        std::filesystem::create_directories(ts_dir);

    std::vector<std::pair<double, double>> timeseries;
    auto *ts_ptr = (save_timeseries != 0) ? &timeseries : nullptr;

    const LaminarOutput out = runLaminar(
        tau,
        k,
        eta,
        theta0,
        dt,
        t_warmup,
        t_laminar,
        record_dt,
        laminar_threshold,
        min_periods,
        seed,
        ts_ptr);

    const std::string slug = paramSlug(tau, k, eta, seed);

    {
        std::ofstream f(summary_dir + "/" + slug + ".tsv");
        f << "tau\tk\teta\ttheta0\tdt\tt_warmup\tt_laminar\trecord_dt\tseed\tlaminar_threshold\tmin_periods"
             "\tn_laminar\tlaminar_mean\tlaminar_median\ttail_alpha\ttail_alpha_se\n";
        f << std::setprecision(10)
          << tau << "\t" << k << "\t" << eta << "\t" << theta0 << "\t" << dt << "\t"
          << t_warmup << "\t" << t_laminar << "\t" << record_dt << "\t" << seed << "\t"
          << laminar_threshold << "\t" << min_periods << "\t"
          << out.n_laminar << "\t" << out.laminar_mean << "\t" << out.laminar_median << "\t"
          << out.tail_alpha << "\t" << out.tail_alpha_se << "\n";
    }

    {
        std::ofstream f(laminar_dir + "/" + slug + ".tsv");
        f << "laminar_length\n";
        for (double v : out.laminar_lengths)
            f << std::setprecision(10) << v << "\n";
    }

    if (save_timeseries != 0)
    {
        std::ofstream f(ts_dir + "/" + slug + ".tsv");
        f << "time\ttheta\n";
        for (const auto &p : timeseries)
            f << std::setprecision(10) << p.first << "\t" << p.second << "\n";
    }

    return 0;
}
