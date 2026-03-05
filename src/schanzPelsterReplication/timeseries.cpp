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

// Full θ(t) timeseries for: dθ/dt = -k·sin(θ(t-τ))
//
// Usage: ./timeseries tau k theta0 dt t_warmup t_measure [record_dt]
// Outputs to:  outputs/timeseries/tau_<τ>_k_<k>_ic_<θ0>_twarmup_<tw>_tmeasure_<tm>_dt_<rdt>.tsv
// Format: time\ttheta  (with header row)

constexpr double DEFAULT_TAU       = 25.0;
constexpr double DEFAULT_K         = 0.165;
constexpr double DEFAULT_THETA0    = 1.5;
constexpr double DEFAULT_DT        = 0.01;
constexpr double DEFAULT_RECORD_DT = 0.1;
constexpr double DEFAULT_T_WARMUP  = 10000.0;
constexpr double DEFAULT_T_MEASURE = 50000.0;

struct HistoryBuffer
{
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta) { times.push_back(t); values.push_back(theta); }

    double getDelayed(double t, double tau, double theta0) const
    {
        double target = t - tau;
        if (times.empty() || target <= times.front()) return theta0;
        if (target >= times.back()) return values.back();
        size_t lo = 0, hi = times.size() - 1;
        while (hi - lo > 1) { size_t mid = (lo + hi) / 2; if (times[mid] <= target) lo = mid; else hi = mid; }
        double a = (target - times[lo]) / (times[hi] - times[lo]);
        return values[lo] + a * (values[hi] - values[lo]);
    }

    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0;
        while (!times.empty() && times.front() < cutoff) { times.pop_front(); values.pop_front(); }
    }
};

double heunStep(HistoryBuffer &hist, double t, double theta,
                double dt, double tau, double theta0, double k)
{
    double td = hist.getDelayed(t, tau, theta0);
    double k1 = -k * std::sin(td);
    double tp = theta + k1 * dt;
    hist.add(t + dt, tp);
    double td2 = hist.getDelayed(t + dt, tau, theta0);
    double k2  = -k * std::sin(td2);
    hist.times.pop_back(); hist.values.pop_back();
    return theta + 0.5 * (k1 + k2) * dt;
}

void simulate(std::ofstream &file,
              double tau, double k, double theta0,
              double dt, double t_warmup, double t_measure, double record_dt)
{
    HistoryBuffer hist;
    double t = 0.0, theta = theta0;
    hist.add(t, theta);

    int warmup_steps = static_cast<int>(t_warmup / dt);
    for (int step = 0; step < warmup_steps; ++step)
    {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);
        if (step % 200 == 0) hist.pruneOld(t, tau);
    }

    int rec_stride    = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    int measure_steps = static_cast<int>(t_measure / dt);

    file << "time\ttheta\n";
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\n";

    for (int step = 0; step < measure_steps; ++step)
    {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);
        if ((step + 1) % rec_stride == 0)
            file << t << "\t" << theta << "\n";
        if (step % 200 == 0) hist.pruneOld(t, tau);
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

    std::ostringstream th0_ss;
    th0_ss << std::fixed << std::setprecision(6) << theta0;
    std::string th0_str = th0_ss.str();
    for (char &c : th0_str) if (c == '-') c = 'n';

    std::ostringstream ss;
    ss << exeDir << "/outputs/timeseries"
       << "/tau_" << tau << "_k_" << k
       << "_ic_" << th0_str
       << "_twarmup_" << t_warmup
       << "_tmeasure_" << t_measure
       << "_dt_" << record_dt << ".tsv";

    std::filesystem::create_directories(
        std::filesystem::path(ss.str()).parent_path());

    std::ofstream file(ss.str());
    simulate(file, tau, k, theta0, dt, t_warmup, t_measure, record_dt);
    file.close();
    return 0;
}
