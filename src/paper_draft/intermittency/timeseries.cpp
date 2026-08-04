#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

// Deterministic DDE:
//   dtheta/dt = -k * sin(theta(t - tau))
//
// Records theta(t) after a warmup, together with the oscillation centre
//   c(t) = (1/T) * int_{t-T}^{t} theta(s) ds,   T = 4*tau,
// which is the slow variable that makes intermittency visible: it sits on a
// plateau during a laminar phase and slides by ~2*pi during a chaotic burst.
//
// Numerics are as in laminar_lengths.cpp: dt divides tau, so the delayed value
// is an exact buffer lookup and the trapezoidal step is explicit.
//
// Usage:
//   ./timeseries tau k theta0 dt t_warmup t_measure record_dt out_tsv
//
// Output columns: time, theta, centre

int main(int argc, char *argv[]) {
    if (argc < 9) {
        std::fprintf(stderr,
                     "Usage: %s tau k theta0 dt t_warmup t_measure record_dt out_tsv\n",
                     argv[0]);
        return 1;
    }

    const double tau = std::atof(argv[1]);
    const double k = std::atof(argv[2]);
    const double theta0 = std::atof(argv[3]);
    const double dt = std::atof(argv[4]);
    const double t_warmup = std::atof(argv[5]);
    const double t_measure = std::atof(argv[6]);
    const double record_dt = std::atof(argv[7]);
    const std::string out_path = argv[8];

    const long n_delay = std::lround(tau / dt);
    const long n_period = std::lround(4.0 * tau / dt);
    const long n_warmup = std::lround(t_warmup / dt);
    const long n_measure = std::lround(t_measure / dt);
    const long stride = std::lround(record_dt / dt);

    std::vector<double> hist(n_delay + 1, theta0);
    std::vector<double> window(n_period, theta0);
    double theta = theta0;
    double window_sum = theta0 * static_cast<double>(n_period);

    const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Could not open output: %s\n", out_path.c_str());
        return 1;
    }

    out << "time\ttheta\tcentre\n";
    out << std::fixed << std::setprecision(6);

    for (long n = 0; n < n_warmup + n_measure; ++n) {
        const double delayed_now = hist[(n + 1) % (n_delay + 1)];
        const double delayed_next = hist[(n + 2) % (n_delay + 1)];
        theta += -0.5 * dt * k * (std::sin(delayed_now) + std::sin(delayed_next));
        hist[(n + 1) % (n_delay + 1)] = theta;

        window_sum += theta - window[n % n_period];
        window[n % n_period] = theta;
        if ((n + 1) % n_period == 0) {
            window_sum = 0.0;
            for (long i = 0; i < n_period; ++i) {
                window_sum += window[i];
            }
        }

        if (n >= n_warmup && (n - n_warmup) % stride == 0) {
            out << static_cast<double>(n - n_warmup) * dt << "\t" << theta << "\t"
                << window_sum / static_cast<double>(n_period) << "\n";
        }
    }
    return 0;
}
