#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
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

constexpr double DEFAULT_K = 0.0;
constexpr double DEFAULT_ETA = 1e-4;
constexpr double DEFAULT_X0 = 0.0;
constexpr double DEFAULT_DT = 0.01;
constexpr double DEFAULT_T_WARMUP = 500.0;
constexpr double DEFAULT_T_LYAP = 2000.0;
constexpr double DEFAULT_RENORM_DT = 1.0;
constexpr double DEFAULT_DELTA0 = 1e-8;
constexpr unsigned int DEFAULT_SEED = 42;
constexpr int DEFAULT_SAVE_TRACE = 0;

struct LyapunovOutput
{
    double lambda = std::numeric_limits<double>::quiet_NaN();
    int n_renorm = 0;
    std::vector<std::pair<double, double>> trace;
};

inline double intermittencyDrift(double x, double k)
{
    // Normal form for type-I intermittency (tangent bifurcation) with saturation term.
    // For small x and k near 0, x' ~ k + x^2; cubic term keeps the trajectory bounded.
    return k + x * x - x * x * x;
}

inline double rk4Step(double x, double dt, double k)
{
    const double k1 = intermittencyDrift(x, k);
    const double k2 = intermittencyDrift(x + 0.5 * dt * k1, k);
    const double k3 = intermittencyDrift(x + 0.5 * dt * k2, k);
    const double k4 = intermittencyDrift(x + dt * k3, k);
    return (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

LyapunovOutput runLyapunov(double k,
                           double eta,
                           double x0,
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

    double t = 0.0;
    double x_base = x0;

    const int warmup_steps = static_cast<int>(std::round(t_warmup / dt));
    for (int step = 0; step < warmup_steps; ++step)
    {
        const double drift = rk4Step(x_base, dt, k);
        const double dW = std::sqrt(dt) * normal(rng);
        x_base += drift + eta * dW;
        t += dt;
    }

    double x_pert = x_base + delta0;

    const int measure_steps = static_cast<int>(std::round(t_lyap / dt));
    const int renorm_steps = std::max(1, static_cast<int>(std::round(renorm_dt / dt)));

    LyapunovOutput out;
    if (save_trace)
        out.trace.reserve(static_cast<size_t>(measure_steps / renorm_steps + 2));

    double log_sum = 0.0;

    for (int step = 0; step < measure_steps; ++step)
    {
        const double dW = std::sqrt(dt) * normal(rng);

        const double drift_base = rk4Step(x_base, dt, k);
        const double drift_pert = rk4Step(x_pert, dt, k);

        x_base += drift_base + eta * dW;
        x_pert += drift_pert + eta * dW;
        t += dt;

        if ((step + 1) % renorm_steps == 0)
        {
            const double dist = std::abs(x_pert - x_base);
            if (std::isfinite(dist) && dist > 0.0)
            {
                log_sum += std::log(dist / delta0);
                ++out.n_renorm;
                if (save_trace)
                    out.trace.emplace_back(t, std::log(dist));
                const double direction = (x_pert > x_base) ? 1.0 : -1.0;
                x_pert = x_base + direction * delta0;
            }
        }
    }

    if (out.n_renorm > 0)
        out.lambda = log_sum / (static_cast<double>(out.n_renorm) * renorm_dt);

    return out;
}

std::string paramSlug(double k, double eta, unsigned int seed)
{
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6);
    oss << "k_" << k << "_eta_" << eta << "_seed_" << seed;
    return oss.str();
}

int main(int argc, char *argv[])
{
    double k = DEFAULT_K;
    double eta = DEFAULT_ETA;
    double x0 = DEFAULT_X0;
    double dt = DEFAULT_DT;
    double t_warmup = DEFAULT_T_WARMUP;
    double t_lyap = DEFAULT_T_LYAP;
    double renorm_dt = DEFAULT_RENORM_DT;
    double delta0 = DEFAULT_DELTA0;
    unsigned int seed = DEFAULT_SEED;
    int save_trace = DEFAULT_SAVE_TRACE;

    if (argc > 1) k = std::stod(argv[1]);
    if (argc > 2) eta = std::stod(argv[2]);
    if (argc > 3) x0 = std::stod(argv[3]);
    if (argc > 4) dt = std::stod(argv[4]);
    if (argc > 5) t_warmup = std::stod(argv[5]);
    if (argc > 6) t_lyap = std::stod(argv[6]);
    if (argc > 7) renorm_dt = std::stod(argv[7]);
    if (argc > 8) delta0 = std::stod(argv[8]);
    if (argc > 9) seed = static_cast<unsigned int>(std::stoul(argv[9]));
    if (argc > 10) save_trace = std::stoi(argv[10]);

    std::string exe_path = argv[0];
    std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();

    const std::string base_out = exe_dir + "/outputs/intermittency_lyapunov"
;
    const std::string summary_dir = base_out + "/run_summaries";
    const std::string trace_dir = base_out + "/traces";

    std::filesystem::create_directories(summary_dir);
    if (save_trace != 0)
        std::filesystem::create_directories(trace_dir);

    const LyapunovOutput out = runLyapunov(
        k,
        eta,
        x0,
        dt,
        t_warmup,
        t_lyap,
        renorm_dt,
        delta0,
        seed,
        save_trace != 0);

    const std::string slug = paramSlug(k, eta, seed);

    {
        std::ofstream f(summary_dir + "/" + slug + ".tsv");
        f << "k\teta\tx0\tdt\tt_warmup\tt_lyap\trenorm_dt\tdelta0\tseed"
             "\tlambda\tn_renorm\n";
        f << std::setprecision(10)
          << k << "\t" << eta << "\t" << x0 << "\t" << dt << "\t"
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
