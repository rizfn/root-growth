#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []() { std::ios_base::sync_with_stdio(false); std::cin.tie(nullptr); return 0; }();

// dθ/dt = -k * f(θ(t-τ)),  f(x) = sign(x) * ln(1+|x|),  τ = 1
// Odd extension of ln(1+x): sublinear growth, natural for x < 0 too.
// Trapezoidal (2nd-order Adams) integration with circular history buffer.
// Usage: ./timeseries_lnp1 k [theta0 t_warmup t_measure record_dt]

constexpr double TAU   = 1.0;
constexpr double DT    = 0.001;
constexpr int    N_BUF = static_cast<int>(TAU / DT) + 1;   // 1001

inline double F(double x) { return std::copysign(std::log1p(std::abs(x)), x); }

void simulate(double k, double theta0, double t_warmup, double t_measure,
              double record_dt, const std::string &out_path)
{
    std::vector<double> buf(N_BUF, theta0);
    int head = 0;
    double theta = theta0;

    long warmup_steps = static_cast<long>(t_warmup / DT);
    for (long s = 0; s < warmup_steps; ++s) {
        double k1 = -k * F(buf[head]);
        double k2 = -k * F(buf[(head + 1) % N_BUF]);
        theta += 0.5 * (k1 + k2) * DT;
        if (!std::isfinite(theta)) break;
        buf[head] = theta;
        head = (head + 1) % N_BUF;
    }

    int  rec_stride    = std::max(1, static_cast<int>(std::round(record_dt / DT)));
    long measure_steps = static_cast<long>(t_measure / DT);

    std::ofstream file(out_path);
    file << "time\ttheta\n" << std::fixed << std::setprecision(6);

    for (long s = 0; s < measure_steps; ++s) {
        double k1 = -k * F(buf[head]);
        double k2 = -k * F(buf[(head + 1) % N_BUF]);
        theta += 0.5 * (k1 + k2) * DT;
        buf[head] = std::isfinite(theta) ? theta : buf[head];
        head = (head + 1) % N_BUF;
        if ((s + 1) % rec_stride == 0)
            file << (t_warmup + (s + 1) * DT) << "\t" << theta << "\n";
        if (!std::isfinite(theta)) break;
    }
}

int main(int argc, char *argv[])
{
    double k         = 2.0;
    double theta0    = 1.0;
    double t_warmup  = 2000.0;
    double t_measure = 3000.0;
    double record_dt = 0.1;

    if (argc > 1) k         = std::stod(argv[1]);
    if (argc > 2) theta0    = std::stod(argv[2]);
    if (argc > 3) t_warmup  = std::stod(argv[3]);
    if (argc > 4) t_measure = std::stod(argv[4]);
    if (argc > 5) record_dt = std::stod(argv[5]);

    std::string dir = std::filesystem::path(argv[0]).parent_path().string();
    std::ostringstream ss;
    ss << dir << "/outputs/lnp1/k_" << k
       << "_ic_" << theta0
       << "_twarmup_" << t_warmup
       << "_tmeasure_" << t_measure << ".tsv";
    std::filesystem::create_directories(std::filesystem::path(ss.str()).parent_path());
    simulate(k, theta0, t_warmup, t_measure, record_dt, ss.str());
    return 0;
}
