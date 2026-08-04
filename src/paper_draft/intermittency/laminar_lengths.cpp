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
// Just above the onset of chaos the attractor is intermittent: the trajectory
// stays on a (nearly) periodic orbit of period 4*tau for a long "laminar"
// stretch, then a short chaotic burst slides it onto a neighbouring orbit
// shifted by ~2*pi.  This program measures the durations of those laminar
// stretches.
//
// The oscillation centre is the running mean of theta over one period,
//   c(t) = (1/T) * int_{t-T}^{t} theta(s) ds,   T = 4*tau,
// which is flat during a laminar phase and slides by ~2*pi during a burst.
//
// Laminar phases are cut out with a two-state detector, so that neither a wobble
// of the centre inside a laminar phase nor a momentary pause inside a burst can
// split a phase in two:
//   laminar -> burst   when the centre leaves its plateau, |c - c_lock| > pi
//                      (half the ~2*pi spacing between neighbouring plateaus);
//   burst -> laminar   once the centre has been quiet for a full period,
//                      |c(t) - c(t - T)| < quiet_threshold.
//
// dt is chosen so that tau/dt and 4*tau/dt are integers: the delayed value is
// then an exact buffer lookup (no interpolation), and the trapezoidal step below
// is explicit because the right-hand side depends only on the past.
//
// Close to the onset the intermittent chaos is only a long transient: the
// trajectory eventually locks onto a stable periodic window and never bursts
// again.  Independent runs (different constant initial histories) are therefore
// pooled, and runs touching either end of the measurement window are flagged as
// censored so that the analysis can drop them.
//
// Usage:
//   ./laminar_lengths tau k theta0_min theta0_max n_runs dt t_warmup t_measure
//                     slip_threshold quiet_threshold out_tsv
//
// Output columns: run, theta0, t_start, length, censored

struct RingBuffer {
    std::vector<double> data;
    long size;

    RingBuffer(long size_, double fill) : data(size_, fill), size(size_) {}

    double &at(long index) { return data[index % size]; }
};

struct LaminarRun {
    int run;
    double theta0;
    double t_start;
    double length;
    bool censored;
};

void measureLaminarPhases(std::vector<LaminarRun> &out,
                          int run_index,
                          double tau,
                          double k,
                          double theta0,
                          double dt,
                          double t_warmup,
                          double t_measure,
                          double slip_threshold,
                          double quiet_threshold) {
    const long n_delay = std::lround(tau / dt);         // steps in one delay
    const long n_period = std::lround(4.0 * tau / dt);  // steps in one period 4*tau
    const long n_warmup = std::lround(t_warmup / dt);
    const long n_measure = std::lround(t_measure / dt);

    // theta_j lives in slot (j mod (n_delay + 1)); the slot holding theta_{j-n_delay}
    // is reused for theta_{j+1} once the delayed value has been consumed.
    RingBuffer hist(n_delay + 1, theta0);
    // Window of the last n_period thetas, and the centres of the last n_period steps.
    RingBuffer window(n_period, theta0);
    RingBuffer centres(n_period + 1, theta0);

    double theta = theta0;
    double window_sum = theta0 * static_cast<double>(n_period);

    bool laminar = true;
    double t_run_start = 0.0;
    double centre_lock = theta0;

    const size_t first_out = out.size();

    for (long n = 0; n < n_warmup + n_measure; ++n) {
        // Trapezoidal step using the two exactly-known delayed values.
        const double delayed_now = hist.at(n + 1);
        const double delayed_next = hist.at(n + 2);
        theta += -0.5 * dt * k * (std::sin(delayed_now) + std::sin(delayed_next));
        hist.at(n + 1) = theta;

        // Rolling mean over one period. The sum is rebuilt once per period so that
        // round-off cannot accumulate over the ~10^9 updates of a long run.
        window_sum += theta - window.at(n);
        window.at(n) = theta;
        if ((n + 1) % n_period == 0) {
            window_sum = 0.0;
            for (long i = 0; i < n_period; ++i) {
                window_sum += window.data[i];
            }
        }
        const double centre = window_sum / static_cast<double>(n_period);
        const double centre_prev = centres.at(n + 2);  // centre one period earlier
        centres.at(n + 1) = centre;

        if (n < n_warmup) {
            continue;
        }

        const double t = static_cast<double>(n - n_warmup) * dt;
        if (n == n_warmup) {
            laminar = true;
            centre_lock = centre;
            t_run_start = t;
            continue;
        }

        if (laminar) {
            if (std::fabs(centre - centre_lock) > slip_threshold) {
                out.push_back({run_index, theta0, t_run_start, t - t_run_start, false});
                laminar = false;
            }
        } else if (std::fabs(centre - centre_prev) < quiet_threshold) {
            laminar = true;
            centre_lock = centre;
            t_run_start = t;
        }
    }

    const double t_end = static_cast<double>(n_measure) * dt;
    if (laminar) {
        out.push_back({run_index, theta0, t_run_start, t_end - t_run_start, true});
    }
    if (out.size() > first_out) {
        out[first_out].censored = true;  // in progress when the window opened
    }
}

int main(int argc, char *argv[]) {
    if (argc < 12) {
        std::fprintf(stderr,
                     "Usage: %s tau k theta0_min theta0_max n_runs dt t_warmup t_measure"
                     " slip_threshold quiet_threshold out_tsv\n",
                     argv[0]);
        return 1;
    }

    const double tau = std::atof(argv[1]);
    const double k = std::atof(argv[2]);
    const double theta0_min = std::atof(argv[3]);
    const double theta0_max = std::atof(argv[4]);
    const int n_runs = std::atoi(argv[5]);
    const double dt = std::atof(argv[6]);
    const double t_warmup = std::atof(argv[7]);
    const double t_measure = std::atof(argv[8]);
    const double slip_threshold = std::atof(argv[9]);
    const double quiet_threshold = std::atof(argv[10]);
    const std::string out_path = argv[11];

    std::vector<LaminarRun> runs;
    for (int i = 0; i < n_runs; ++i) {
        const double frac = (n_runs > 1) ? static_cast<double>(i) / (n_runs - 1) : 0.0;
        const double theta0 = theta0_min + frac * (theta0_max - theta0_min);
        measureLaminarPhases(runs, i, tau, k, theta0, dt, t_warmup, t_measure,
                             slip_threshold, quiet_threshold);
    }

    const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Could not open output: %s\n", out_path.c_str());
        return 1;
    }

    out << "run\ttheta0\tt_start\tlength\tcensored\n";
    out << std::fixed << std::setprecision(4);
    for (const LaminarRun &r : runs) {
        out << r.run << "\t" << r.theta0 << "\t" << r.t_start << "\t" << r.length
            << "\t" << (r.censored ? 1 : 0) << "\n";
    }
    return 0;
}
