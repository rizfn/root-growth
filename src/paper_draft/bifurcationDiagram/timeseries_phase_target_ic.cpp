#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// Deterministic DDE:
//   dtheta/dt = -k * sin(theta(t - tau))
//
// Two-stage branch-guided workflow:
// 1) warmup at k_target (pick branch / basin)
// 2) warmup at k_real
// 3) record only the final short segment at k_real
//
// Usage:
//   ./timeseries_phase_target_ic tau k_real k_target theta0 dt
//       t_warmup_target t_warmup_real t_record record_dt out_tsv

struct HistoryBuffer {
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta) {
        times.push_back(t);
        values.push_back(theta);
    }

    void pop_back() {
        if (!times.empty()) {
            times.pop_back();
            values.pop_back();
        }
    }

    double getDelayed(double t, double tau, double theta0) const {
        const double target = t - tau;
        if (times.empty() || target <= times.front()) {
            return theta0;
        }
        if (target >= times.back()) {
            return values.back();
        }

        size_t lo = 0;
        size_t hi = times.size() - 1;
        while (hi - lo > 1) {
            const size_t mid = (lo + hi) / 2;
            if (times[mid] <= target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        const double dt = times[hi] - times[lo];
        if (dt <= 0.0) {
            return values[lo];
        }
        const double alpha = (target - times[lo]) / dt;
        return values[lo] + alpha * (values[hi] - values[lo]);
    }

    void pruneOld(double current_time, double tau) {
        const double cutoff = current_time - tau - 2.0;
        while (!times.empty() && times.front() < cutoff) {
            times.pop_front();
            values.pop_front();
        }
    }
};

static inline double rhs(
    HistoryBuffer &hist,
    double t_eval,
    double theta_eval,
    double tau,
    double theta0,
    double k
) {
    // Temporarily append stage state so delayed interpolation can use it.
    hist.add(t_eval, theta_eval);
    const double theta_delay = hist.getDelayed(t_eval, tau, theta0);
    hist.pop_back();
    return -k * std::sin(theta_delay);
}

static inline double rk4Step(
    HistoryBuffer &hist,
    double t,
    double theta,
    double dt,
    double tau,
    double theta0,
    double k
) {
    const double k1 = rhs(hist, t, theta, tau, theta0, k);
    const double k2 = rhs(hist, t + 0.5 * dt, theta + 0.5 * dt * k1, tau, theta0, k);
    const double k3 = rhs(hist, t + 0.5 * dt, theta + 0.5 * dt * k2, tau, theta0, k);
    const double k4 = rhs(hist, t + dt, theta + dt * k3, tau, theta0, k);
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

void integrateSilent(
    HistoryBuffer &hist,
    double &t,
    double &theta,
    double tau,
    double k,
    double theta0,
    double dt,
    double duration
) {
    const int n_steps = std::max(0, static_cast<int>(std::round(duration / dt)));
    for (int step = 0; step < n_steps; ++step) {
        theta = rk4Step(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);
        if ((step % 400) == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

void integrateAndRecord(
    std::ofstream &out,
    HistoryBuffer &hist,
    double &t,
    double &theta,
    double tau,
    double k,
    double theta0,
    double dt,
    double t_record,
    double record_dt
) {
    const int rec_stride = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    const int n_steps = std::max(0, static_cast<int>(std::round(t_record / dt)));

    out << "time\ttheta\n";
    out << std::fixed << std::setprecision(8) << t << "\t" << theta << "\n";

    for (int step = 0; step < n_steps; ++step) {
        theta = rk4Step(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);

        if (((step + 1) % rec_stride) == 0) {
            out << t << "\t" << theta << "\n";
        }
        if ((step % 400) == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 11) {
        std::cerr
            << "Usage: " << argv[0]
            << " tau k_real k_target theta0 dt"
            << " t_warmup_target t_warmup_real t_record record_dt out_tsv\n";
        return 1;
    }

    const double tau = std::stod(argv[1]);
    const double k_real = std::stod(argv[2]);
    const double k_target = std::stod(argv[3]);
    const double theta0 = std::stod(argv[4]);
    const double dt = std::stod(argv[5]);
    const double t_warmup_target = std::stod(argv[6]);
    const double t_warmup_real = std::stod(argv[7]);
    const double t_record = std::stod(argv[8]);
    const double record_dt = std::stod(argv[9]);
    const std::string out_path = argv[10];

    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << out_path << "\n";
        return 2;
    }

    HistoryBuffer hist;
    double t = 0.0;
    double theta = theta0;

    // Constant-history prefill for t < 0.
    const int n_init = std::max(1, static_cast<int>(std::round(tau / dt)));
    for (int i = n_init; i >= 1; --i) {
        hist.times.push_back(-static_cast<double>(i) * dt);
        hist.values.push_back(theta0);
    }
    hist.add(t, theta);

    integrateSilent(hist, t, theta, tau, k_target, theta0, dt, t_warmup_target);
    integrateSilent(hist, t, theta, tau, k_real, theta0, dt, t_warmup_real);
    integrateAndRecord(out, hist, t, theta, tau, k_real, theta0, dt, t_record, record_dt);
    return 0;
}
