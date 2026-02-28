#include <vector>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <array>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

// Deterministic DDE: dθ/dt = -k·sin(θ(t-τ))
// Lyapunov exponent estimation via perturbation method.
// Strategy: run noiseHeun warmup to settle on attractor, then perturb the
// ENTIRE history segment θ(s), s∈[t-τ, t] by a constant delta.
// For DDEs the state is a function (not a point), so perturbing all history
// values gives a well-defined perturbation of the full infinite-dimensional
// state — unlike perturbing only the last point, which leaves the delay term
// unperturbed for the first τ time units.

constexpr double DEFAULT_TAU       = 25.0;    // Time lag
constexpr double DEFAULT_K         = 0.21;    // Gravitropic strength
constexpr double DEFAULT_THETA0    = 1.5708;  // Initial angle (π/2)
constexpr double DEFAULT_DT        = 0.01;    // Integration step
constexpr double DEFAULT_RECORD_DT = 0.1;     // Recording interval
constexpr double DEFAULT_T_WARMUP  = 2000.0;  // Warmup to reach attractor
constexpr double DEFAULT_T_LYAP    = 5000.0;  // Measurement window

// Perturbation magnitudes (added uniformly to all history values)
constexpr int    N_PERTURB = 5;
constexpr double DELTAS[N_PERTURB] = {1e-6, 1e-7, 1e-8, 1e-9, 1e-10};

// History buffer to store past theta values
struct HistoryBuffer
{
    std::deque<double> times;
    std::deque<double> values;

    void add(double t, double theta)
    {
        times.push_back(t);
        values.push_back(theta);
    }

    // Get theta at time (t - tau) using linear interpolation
    double getDelayed(double t, double tau, double theta0) const
    {
        double target_time = t - tau;

        // Before recorded history: return constant initial condition
        if (times.empty() || target_time <= times.front())
            return theta0;

        // After last point (shouldn't happen in normal use)
        if (target_time >= times.back())
            return values.back();

        // Find bracket and interpolate
        for (size_t i = 0; i < times.size() - 1; ++i)
        {
            if (times[i] <= target_time && target_time <= times[i + 1])
            {
                double t1 = times[i],  t2 = times[i + 1];
                double v1 = values[i], v2 = values[i + 1];
                double alpha = (target_time - t1) / (t2 - t1);
                return v1 + alpha * (v2 - v1);
            }
        }

        return values.back();
    }

    // Remove old history to save memory (keep only what's needed for delay)
    void pruneOld(double current_time, double tau)
    {
        double cutoff = current_time - tau - 1.0; // Keep a bit extra for safety
        while (!times.empty() && times.front() < cutoff)
        {
            times.pop_front();
            values.pop_front();
        }
    }
};

// One deterministic Heun step: updates history in-place, returns new theta
double heunStep(HistoryBuffer &history, double t, double theta, double dt, double tau, double theta0, double k)
{
    // Predictor: derivative at current delayed state
    double theta_delayed = history.getDelayed(t, tau, theta0);
    double k1 = -k * std::sin(theta_delayed);
    double theta_pred = theta + k1 * dt;
    double t_next = t + dt;

    // Temporarily insert predictor into history for corrector delay lookup
    history.add(t_next, theta_pred);

    double theta_delayed_pred = history.getDelayed(t_next, tau, theta0);
    double k2 = -k * std::sin(theta_delayed_pred);

    // Remove temporary predictor
    history.times.pop_back();
    history.values.pop_back();

    // Corrector: Heun average
    return theta + 0.5 * (k1 + k2) * dt;
}

void solveLyapunov(std::ofstream &file,
                   double tau, double k, double theta0,
                   double dt, double t_warmup, double t_lyap, double record_dt)
{
    // ── Warmup phase: settle onto attractor ─────────────────────
    HistoryBuffer warmup_history;
    double t     = 0.0;
    double theta = theta0;
    warmup_history.add(t, theta);

    int warmup_steps = static_cast<int>(t_warmup / dt);
    for (int step = 0; step < warmup_steps; ++step)
    {
        theta = heunStep(warmup_history, t, theta, dt, tau, theta0, k);
        t += dt;
        warmup_history.add(t, theta);
        if (step % 100 == 0)
            warmup_history.pruneOld(t, tau);
    }

    // ── Build baseline + perturbed trajectories ──────────────────
    // Total trajectories: 1 baseline + N_PERTURB perturbed
    // Each perturbed copy has ALL history values shifted by +delta.
    constexpr int N_TRAJ = 1 + N_PERTURB;
    std::array<HistoryBuffer, N_TRAJ> histories;
    std::array<double, N_TRAJ>        thetas;

    histories[0] = warmup_history;
    thetas[0]    = theta;

    for (int p = 0; p < N_PERTURB; ++p)
    {
        histories[p + 1] = warmup_history;
        for (auto &v : histories[p + 1].values)
            v += DELTAS[p];
        thetas[p + 1] = theta + DELTAS[p];
    }

    // ── Lyapunov measurement phase ───────────────────────────────
    int record_steps  = std::max(1, static_cast<int>(std::round(record_dt / dt)));
    int lyapunov_steps = static_cast<int>(t_lyap / dt);

    // Header: time + baseline + one column per perturbation delta
    file << "time\ttheta_baseline";
    for (int p = 0; p < N_PERTURB; ++p)
        file << "\ttheta_delta_" << std::scientific << std::setprecision(0) << DELTAS[p];
    file << "\n";

    // Write initial (post-warmup) state
    file << std::fixed << std::setprecision(6) << t;
    for (int i = 0; i < N_TRAJ; ++i)
        file << "\t" << thetas[i];
    file << "\n";

    for (int step = 0; step < lyapunov_steps; ++step)
    {
        for (int i = 0; i < N_TRAJ; ++i)
        {
            thetas[i] = heunStep(histories[i], t, thetas[i], dt, tau, theta0, k);
            histories[i].add(t + dt, thetas[i]);
        }
        t += dt;

        if ((step + 1) % record_steps == 0)
        {
            file << t;
            for (int i = 0; i < N_TRAJ; ++i)
                file << "\t" << thetas[i];
            file << "\n";
        }

        if (step % 100 == 0)
            for (int i = 0; i < N_TRAJ; ++i)
                histories[i].pruneOld(t, tau);

    }
}

int main(int argc, char *argv[])
{
    double tau       = DEFAULT_TAU;
    double k         = DEFAULT_K;
    double theta0    = DEFAULT_THETA0;
    double dt        = DEFAULT_DT;
    double t_warmup  = DEFAULT_T_WARMUP;
    double t_lyap    = DEFAULT_T_LYAP;
    double record_dt = DEFAULT_RECORD_DT;

    if (argc > 1) tau       = std::stod(argv[1]);
    if (argc > 2) k         = std::stod(argv[2]);
    if (argc > 3) theta0    = std::stod(argv[3]);
    if (argc > 4) dt        = std::stod(argv[4]);
    if (argc > 5) t_warmup  = std::stod(argv[5]);
    if (argc > 6) t_lyap    = std::stod(argv[6]);
    if (argc > 7) record_dt = std::stod(argv[7]);

    std::string exePath = argv[0];
    std::string exeDir  = std::filesystem::path(exePath).parent_path().string();

    std::ostringstream filePathStream;
    filePathStream << exeDir << "/outputs/lyapunov/k_sweep"
                   << "/tau_" << tau << "_k_" << k << "_theta0_" << theta0
                   << "_twarmup_" << t_warmup << "_tlyap_" << t_lyap
                   << "_dt_" << record_dt << ".tsv";
    std::string filePath = filePathStream.str();

    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file(filePath);
    solveLyapunov(file, tau, k, theta0, dt, t_warmup, t_lyap, record_dt);
    file.close();

    return 0;
}