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

// Deterministic DDE: dθ/dt = -k·sin(θ(t-τ))
//
// Outputs Poincaré section events after warmup, for bifurcation diagrams.
// Three event types recorded per line:
//   LOCAL_MAX  : local maximum of θ(t)  → θ value
//   LOCAL_MIN  : local minimum of θ(t)  → θ value
//   ZERO_UP    : upward zero crossing    → dθ/dt at crossing (interpolated)
//   ZERO_DOWN  : downward zero crossing  → dθ/dt at crossing (interpolated)
//
// Usage:
//   ./bifurcation tau k theta0 dt t_warmup t_measure [record_dt]
//
// record_dt controls the density of the internal record buffer used for
// Poincaré detection; it does NOT affect numerical integration (dt does).

constexpr double DEFAULT_TAU       = 25.0;
constexpr double DEFAULT_K         = 0.165;
constexpr double DEFAULT_THETA0    = 1.5708;
constexpr double DEFAULT_DT        = 0.01;
constexpr double DEFAULT_RECORD_DT = 0.05;
constexpr double DEFAULT_T_WARMUP  = 10000.0;
constexpr double DEFAULT_T_MEASURE = 50000.0;

// ── History buffer ──────────────────────────────────────────────────────────
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
        double target = t - tau;
        if (times.empty() || target <= times.front()) return theta0;
        if (target >= times.back())  return values.back();
        // Binary search for bracket
        size_t lo = 0, hi = times.size() - 1;
        while (hi - lo > 1)
        {
            size_t mid = (lo + hi) / 2;
            if (times[mid] <= target) lo = mid; else hi = mid;
        }
        double alpha = (target - times[lo]) / (times[hi] - times[lo]);
        return values[lo] + alpha * (values[hi] - values[lo]);
    }

    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0;
        while (!times.empty() && times.front() < cutoff)
        {
            times.pop_front();
            values.pop_front();
        }
    }
};

// ── Heun (RK2) integrator step ───────────────────────────────────────────────
double heunStep(HistoryBuffer &hist, double t, double theta,
                double dt, double tau, double theta0, double k)
{
    double theta_d = hist.getDelayed(t, tau, theta0);
    double k1      = -k * std::sin(theta_d);
    double theta_p = theta + k1 * dt;

    hist.add(t + dt, theta_p);
    double theta_d2 = hist.getDelayed(t + dt, tau, theta0);
    double k2       = -k * std::sin(theta_d2);
    hist.times.pop_back();
    hist.values.pop_back();

    return theta + 0.5 * (k1 + k2) * dt;
}

// ── Poincaré recorder ────────────────────────────────────────────────────────
// Keeps a sliding window of (t, theta) and emits events.
struct PoincaréRecorder
{
    std::ofstream &out;
    // Rolling window of three consecutive points for extremum detection
    double t_prev = 0, th_prev = 0;
    double t_cur  = 0, th_cur  = 0;
    bool   initialised = false;

    explicit PoincaréRecorder(std::ofstream &f) : out(f) {}

    void push(double t, double theta)
    {
        if (!initialised)
        {
            t_prev = t; th_prev = theta;
            initialised = true;
            return;
        }
        // We have two points; need one more to detect extrema
        // Detect using t_prev → t_cur → t (three points)
        if (t_cur == 0 && t_prev != 0)
        {
            t_cur = t; th_cur = theta;
            return;
        }

        double t_next = t, th_next = theta;

        // ── Local maximum ────────────────────────────────────────────
        if (th_cur > th_prev && th_cur > th_next)
            out << "LOCAL_MAX\t" << std::fixed << std::setprecision(8)
                << t_cur << "\t" << th_cur << "\n";

        // ── Local minimum ────────────────────────────────────────────
        if (th_cur < th_prev && th_cur < th_next)
            out << "LOCAL_MIN\t" << std::fixed << std::setprecision(8)
                << t_cur << "\t" << th_cur << "\n";

        // ── Zero crossings (between prev and cur) ────────────────────
        // Upward: th_prev < 0 and th_cur >= 0
        if (th_prev < 0.0 && th_cur >= 0.0)
        {
            // Linear interpolation to find crossing time
            double frac  = -th_prev / (th_cur - th_prev);
            double t_x   = t_prev + frac * (t_cur - t_prev);
            double dth   = (th_cur - th_prev) / (t_cur - t_prev);  // slope ≈ dθ/dt
            out << "ZERO_UP\t" << std::fixed << std::setprecision(8)
                << t_x << "\t" << dth << "\n";
        }
        // Downward: th_prev > 0 and th_cur <= 0
        if (th_prev > 0.0 && th_cur <= 0.0)
        {
            double frac  = th_prev / (th_prev - th_cur);
            double t_x   = t_prev + frac * (t_cur - t_prev);
            double dth   = (th_cur - th_prev) / (t_cur - t_prev);
            out << "ZERO_DOWN\t" << std::fixed << std::setprecision(8)
                << t_x << "\t" << dth << "\n";
        }

        // Advance window
        t_prev = t_cur;  th_prev = th_cur;
        t_cur  = t_next; th_cur  = th_next;
    }
};

// ── Main simulation ──────────────────────────────────────────────────────────
void simulate(std::ofstream &file,
              double tau, double k, double theta0,
              double dt, double t_warmup, double t_measure,
              double record_dt)
{
    HistoryBuffer hist;
    double t = 0.0, theta = theta0;
    hist.add(t, theta);

    // ── Warmup (silent) ──────────────────────────────────────────────────────
    int warmup_steps = static_cast<int>(t_warmup / dt);
    for (int step = 0; step < warmup_steps; ++step)
    {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t    += dt;
        hist.add(t, theta);
        if (step % 200 == 0) hist.pruneOld(t, tau);
    }

    // ── Measurement phase ────────────────────────────────────────────────────
    int rec_stride    = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    int measure_steps = static_cast<int>(t_measure / dt);

    file << "type\ttime\tvalue\n";

    PoincaréRecorder rec(file);
    rec.push(t, theta);  // seed with post-warmup state

    for (int step = 0; step < measure_steps; ++step)
    {
        theta = heunStep(hist, t, theta, dt, tau, theta0, k);
        t    += dt;
        hist.add(t, theta);

        if ((step + 1) % rec_stride == 0)
            rec.push(t, theta);

        if (step % 200 == 0) hist.pruneOld(t, tau);
    }
}

// ── entry point ─────────────────────────────────────────────────────────────
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

    // Encode theta0 to avoid sign issues in filename
    std::ostringstream th0_ss;
    th0_ss << std::fixed << std::setprecision(6) << theta0;
    std::string th0_str = th0_ss.str();
    // Replace '-' with 'n' for negative ICs
    for (char &c : th0_str) if (c == '-') c = 'n';

    std::ostringstream ss;
    ss << exeDir << "/outputs/bifurcation"
       << "/tau_" << tau << "_k_" << k
       << "_ic_" << th0_str
       << "_twarmup_" << t_warmup
       << "_tmeasure_" << t_measure
       << "_dt_" << record_dt << ".tsv";
    std::string filePath = ss.str();

    std::filesystem::create_directories(
        std::filesystem::path(filePath).parent_path());

    std::ofstream file(filePath);
    simulate(file, tau, k, theta0, dt, t_warmup, t_measure, record_dt);
    file.close();

    return 0;
}
