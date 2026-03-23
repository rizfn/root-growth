#include <algorithm>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Deterministic DDE:
//   dtheta/dt = -k * sin(theta(t - tau))
//
// Workflow:
// 1) Warm up at target_k to construct a branch-specific initial condition.
// 2) Switch to real_k and run warmup + measurement.
// 3) Write local maxima (Poincare-style section) during measurement.
//
// Usage:
//   ./bifurcation_target_ic tau real_k target_k theta0 dt
//       t_warmup_target t_warmup_real t_measure record_dt out_tsv
//
// Output columns:
//   tauk_real, k_real, tauk_target, time, theta_max

constexpr double DEFAULT_TAU = 25.0;
constexpr double DEFAULT_REAL_K = 0.16;
constexpr double DEFAULT_TARGET_K = 0.16;
constexpr double DEFAULT_THETA0 = 1.0;
constexpr double DEFAULT_DT = 0.01;
constexpr double DEFAULT_WARMUP_TARGET = 2000.0;
constexpr double DEFAULT_WARMUP_REAL = 10000.0;
constexpr double DEFAULT_MEASURE = 30000.0;
constexpr double DEFAULT_RECORD_DT = 0.05;

struct HistoryBuffer {
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta) {
        times.push_back(t);
        values.push_back(theta);
    }

    double getDelayed(double t, double tau) const {
        const double target = t - tau;
        if (times.empty()) {
            return 0.0;
        }
        if (target <= times.front()) {
            return values.front();
        }
        if (target >= times.back()) {
            return values.back();
        }

        size_t lo = 0;
        size_t hi = times.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
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

double heunStep(HistoryBuffer &hist, double t, double theta, double dt, double tau, double k) {
    const double td1 = hist.getDelayed(t, tau);
    const double k1 = -k * std::sin(td1);
    const double theta_pred = theta + k1 * dt;

    hist.add(t + dt, theta_pred);
    const double td2 = hist.getDelayed(t + dt, tau);
    const double k2 = -k * std::sin(td2);
    hist.times.pop_back();
    hist.values.pop_back();

    return theta + 0.5 * (k1 + k2) * dt;
}

struct MaxRecorder {
    std::ofstream &out;
    const double tau;
    const double real_k;
    const double target_k;

    bool seeded = false;
    bool has_cur = false;
    double t_prev = 0.0;
    double th_prev = 0.0;
    double t_cur = 0.0;
    double th_cur = 0.0;

    MaxRecorder(std::ofstream &f, double tau_, double real_k_, double target_k_)
        : out(f), tau(tau_), real_k(real_k_), target_k(target_k_) {}

    void push(double t, double theta) {
        if (!seeded) {
            t_prev = t;
            th_prev = theta;
            seeded = true;
            return;
        }
        if (!has_cur) {
            t_cur = t;
            th_cur = theta;
            has_cur = true;
            return;
        }

        if (th_cur > th_prev && th_cur > theta) {
            out << std::fixed << std::setprecision(8)
                << (tau * real_k) << "\t"
                << real_k << "\t"
                << (tau * target_k) << "\t"
                << t_cur << "\t"
                << th_cur << "\n";
        }

        t_prev = t_cur;
        th_prev = th_cur;
        t_cur = t;
        th_cur = theta;
    }
};

void runSegment(HistoryBuffer &hist, double &t, double &theta, double tau, double k, double dt, double duration) {
    const int n_steps = std::max(0, static_cast<int>(std::round(duration / dt)));
    for (int step = 0; step < n_steps; ++step) {
        theta = heunStep(hist, t, theta, dt, tau, k);
        t += dt;
        hist.add(t, theta);
        if (step % 200 == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

void simulate(std::ofstream &out,
              double tau,
              double real_k,
              double target_k,
              double theta0,
              double dt,
              double warmup_target,
              double warmup_real,
              double t_measure,
              double record_dt) {
    HistoryBuffer hist;
    double t = 0.0;
    double theta = theta0;
    hist.add(t, theta);

    // Ensure delayed interpolation has sufficient prehistory before t=0.
    const int n_init = std::max(1, static_cast<int>(std::round(tau / dt)));
    for (int i = n_init; i >= 1; --i) {
        const double ti = -i * dt;
        hist.times.push_front(ti);
        hist.values.push_front(theta0);
    }

    // Branch-selective target warmup.
    runSegment(hist, t, theta, tau, target_k, dt, warmup_target);

    // Real-parameter warmup before recording section events.
    runSegment(hist, t, theta, tau, real_k, dt, warmup_real);

    const int stride = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    const int meas_steps = std::max(0, static_cast<int>(std::round(t_measure / dt)));

    out << "tauk_real\tk_real\ttauk_target\ttime\ttheta_max\n";

    MaxRecorder recorder(out, tau, real_k, target_k);
    recorder.push(t, theta);

    for (int step = 0; step < meas_steps; ++step) {
        theta = heunStep(hist, t, theta, dt, tau, real_k);
        t += dt;
        hist.add(t, theta);

        if ((step + 1) % stride == 0) {
            recorder.push(t, theta);
        }
        if (step % 200 == 0) {
            hist.pruneOld(t, tau);
        }
    }
}

int main(int argc, char *argv[]) {
    double tau = DEFAULT_TAU;
    double real_k = DEFAULT_REAL_K;
    double target_k = DEFAULT_TARGET_K;
    double theta0 = DEFAULT_THETA0;
    double dt = DEFAULT_DT;
    double warmup_target = DEFAULT_WARMUP_TARGET;
    double warmup_real = DEFAULT_WARMUP_REAL;
    double t_measure = DEFAULT_MEASURE;
    double record_dt = DEFAULT_RECORD_DT;
    std::string out_path;

    if (argc >= 11) {
        tau = std::stod(argv[1]);
        real_k = std::stod(argv[2]);
        target_k = std::stod(argv[3]);
        theta0 = std::stod(argv[4]);
        dt = std::stod(argv[5]);
        warmup_target = std::stod(argv[6]);
        warmup_real = std::stod(argv[7]);
        t_measure = std::stod(argv[8]);
        record_dt = std::stod(argv[9]);
        out_path = argv[10];
    } else {
        std::cerr << "Usage: " << argv[0]
                  << " tau real_k target_k theta0 dt"
                  << " t_warmup_target t_warmup_real t_measure record_dt out_tsv"
                  << std::endl;
        return 1;
    }

    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "Could not open output: " << out_path << std::endl;
        return 1;
    }

    simulate(out, tau, real_k, target_k, theta0, dt, warmup_target, warmup_real, t_measure, record_dt);
    return 0;
}
