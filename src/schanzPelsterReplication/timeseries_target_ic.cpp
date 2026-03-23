#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

// Two-stage warmup timeseries solver for:
//   dtheta/dt = -k * sin(theta(t-tau))
//
// Designed to mirror the target-IC workflow used by bifurcation_target_ic scripts:
//   1) warmup at k_target (guides trajectory into desired basin)
//   2) warmup at k_real
//   3) measure at k_real and write full timeseries
//
// Usage:
//   ./timeseries_target_ic tau k_real k_target theta0 dt
//                         t_warmup_target t_warmup_real t_measure record_dt out_file

struct HistoryBuffer {
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta) {
        times.push_back(t);
        values.push_back(theta);
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
        const double alpha = (target - times[lo]) / (times[hi] - times[lo]);
        return values[lo] + alpha * (values[hi] - values[lo]);
    }

    void pruneOld(double current_time, double tau) {
        const double cutoff = current_time - tau - 1.0;
        while (!times.empty() && times.front() < cutoff) {
            times.pop_front();
            values.pop_front();
        }
    }
};

double heunStep(
    HistoryBuffer &hist,
    double t,
    double theta,
    double dt,
    double tau,
    double theta0,
    double k
) {
    const double td = hist.getDelayed(t, tau, theta0);
    const double k1 = -k * std::sin(td);
    const double theta_pred = theta + k1 * dt;

    hist.add(t + dt, theta_pred);
    const double td2 = hist.getDelayed(t + dt, tau, theta0);
    const double k2 = -k * std::sin(td2);
    hist.times.pop_back();
    hist.values.pop_back();

    return theta + 0.5 * (k1 + k2) * dt;
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
    const int n_steps = static_cast<int>(duration / dt);
    for (int step = 0; step < n_steps; ++step) {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);
        if (step % 200 == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

void integrateAndRecord(
    std::ofstream &file,
    HistoryBuffer &hist,
    double &t,
    double &theta,
    double tau,
    double k,
    double theta0,
    double dt,
    double measure,
    double record_dt
) {
    const int rec_stride = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    const int measure_steps = static_cast<int>(measure / dt);

    file << "time\ttheta\n";
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\n";

    for (int step = 0; step < measure_steps; ++step) {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t += dt;
        hist.add(t, theta);

        if ((step + 1) % rec_stride == 0) {
            file << t << "\t" << theta << "\n";
        }
        if (step % 200 == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 11) {
        std::cerr
            << "Usage: " << argv[0]
            << " tau k_real k_target theta0 dt t_warmup_target"
            << " t_warmup_real t_measure record_dt out_file\n";
        return 1;
    }

    const double tau = std::stod(argv[1]);
    const double k_real = std::stod(argv[2]);
    const double k_target = std::stod(argv[3]);
    const double theta0 = std::stod(argv[4]);
    const double dt = std::stod(argv[5]);
    const double t_warmup_target = std::stod(argv[6]);
    const double t_warmup_real = std::stod(argv[7]);
    const double t_measure = std::stod(argv[8]);
    const double record_dt = std::stod(argv[9]);
    const std::string out_file = argv[10];

    std::filesystem::create_directories(std::filesystem::path(out_file).parent_path());
    std::ofstream file(out_file);
    if (!file) {
        std::cerr << "Failed to open output file: " << out_file << "\n";
        return 2;
    }

    HistoryBuffer hist;
    double t = 0.0;
    double theta = theta0;
    hist.add(t, theta);

    integrateSilent(hist, t, theta, tau, k_target, theta0, dt, t_warmup_target);
    integrateSilent(hist, t, theta, tau, k_real, theta0, dt, t_warmup_real);
    integrateAndRecord(file, hist, t, theta, tau, k_real, theta0, dt, t_measure, record_dt);

    return 0;
}
