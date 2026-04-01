#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

// Deterministic DDE:
//   dtheta/dt = -k * sin(theta(t - tau))
//
// This executable runs a full scan around a critical point k_c and measures
// ghost escape times using a sliding-window criterion:
//   first t such that max(theta over [t-win_tau, t]) < pi
//
// Two modes are supported:
// - arrival:        scan below k_c (k = k_c - delta)
// - disappearance:  scan above k_c (k = k_c + delta)
//
// Output files:
// - samples TSV: one line per (k, sample)
// - summary TSV: one line per k

namespace {

struct Config {
    std::string mode;
    double tau = 1.0;
    double dt = 0.01;

    double k_ic = 4.13;
    double k_c = 3.9849;

    double t_warmup = 50000.0;
    double t_max_tau = 10000.0;
    double win_tau = 6.0;

    int n_samples = 40;
    double spacing_tau = 1.0;

    int n_delta = 36;
    double delta_min = 1e-4;
    double delta_max = 0.20;

    std::string out_samples;
    std::string out_summary;
};

struct SampleResult {
    double k = 0.0;
    double delta = 0.0;
    int sample_id = 0;
    double t_esc_tau = 0.0;
    int censored = 0;
};

struct SummaryRow {
    double k = 0.0;
    double delta = 0.0;
    int n_samples = 0;
    int n_escaped = 0;
    int n_censored = 0;
    double t_mean_escaped = std::numeric_limits<double>::quiet_NaN();
    double t_median_escaped = std::numeric_limits<double>::quiet_NaN();
    double t_p25_escaped = std::numeric_limits<double>::quiet_NaN();
    double t_p75_escaped = std::numeric_limits<double>::quiet_NaN();
};

bool isFinite(double x) {
    return std::isfinite(x);
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

double drift(double theta_delay, double k) {
    return -k * std::sin(theta_delay);
}

std::vector<double> spinupDDE(double tau, double dt, double k, double theta0, double t_warmup) {
    const int n_buf = std::max(2, static_cast<int>(std::llround(tau / dt)) + 1);
    std::vector<double> buf(static_cast<size_t>(n_buf), theta0);
    double theta = theta0;

    const int n_steps = std::max(0, static_cast<int>(std::llround(t_warmup / dt)));
    for (int i = 0; i < n_steps; ++i) {
        const double k1 = drift(buf.front(), k);
        const double k2 = drift(buf[1], k);
        theta += 0.5 * (k1 + k2) * dt;

        std::rotate(buf.begin(), buf.begin() + 1, buf.end());
        buf.back() = theta;
    }

    return buf;
}

std::vector<std::vector<double>> collectICs(
    double tau,
    double dt,
    double k,
    double t_warmup,
    int n_samples,
    double spacing_tau,
    double theta0
) {
    std::vector<std::vector<double>> samples;
    samples.reserve(static_cast<size_t>(n_samples));

    std::vector<double> buf = spinupDDE(tau, dt, k, theta0, t_warmup);
    double theta = buf.back();

    const int spacing_steps = std::max(1, static_cast<int>(std::llround((spacing_tau * tau) / dt)));

    for (int i = 0; i < n_samples; ++i) {
        samples.push_back(buf);
        for (int s = 0; s < spacing_steps; ++s) {
            const double k1 = drift(buf.front(), k);
            const double k2 = drift(buf[1], k);
            theta += 0.5 * (k1 + k2) * dt;
            std::rotate(buf.begin(), buf.begin() + 1, buf.end());
            buf.back() = theta;
        }
    }

    return samples;
}

double escapeTimeTau(
    double tau,
    double dt,
    double k,
    const std::vector<double>& buf_init,
    double win_tau,
    double t_max_tau
) {
    std::vector<double> buf = buf_init;
    double theta = buf.back();

    const int win_steps = std::max(1, static_cast<int>(std::llround((win_tau * tau) / dt)));
    const int t_max_steps = std::max(1, static_cast<int>(std::llround((t_max_tau * tau) / dt)));

    std::deque<std::pair<int, double>> maxq;

    for (int step = 0; step < t_max_steps; ++step) {
        const double k1 = drift(buf.front(), k);
        const double k2 = drift(buf[1], k);
        theta += 0.5 * (k1 + k2) * dt;

        std::rotate(buf.begin(), buf.begin() + 1, buf.end());
        buf.back() = theta;

        while (!maxq.empty() && maxq.back().second <= theta) {
            maxq.pop_back();
        }
        maxq.emplace_back(step, theta);

        const int oldest = step - win_steps + 1;
        while (!maxq.empty() && maxq.front().first < oldest) {
            maxq.pop_front();
        }

        if (step >= win_steps - 1) {
            if (!maxq.empty() && maxq.front().second < M_PI) {
                const double t_escape = (static_cast<double>(step) + 1.0) * dt;
                return t_escape / tau;
            }
        }
    }

    return t_max_tau;
}

std::vector<double> geomspace(double start, double stop, int n) {
    std::vector<double> out;
    if (n <= 0) {
        return out;
    }
    out.reserve(static_cast<size_t>(n));

    if (n == 1) {
        out.push_back(start);
        return out;
    }

    const double a = std::log(start);
    const double b = std::log(stop);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        out.push_back(std::exp((1.0 - t) * a + t * b));
    }
    return out;
}

double quantileSorted(const std::vector<double>& sorted_vals, double q) {
    if (sorted_vals.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double idx = q * static_cast<double>(sorted_vals.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(idx));
    const auto hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) {
        return sorted_vals[lo];
    }
    const double w = idx - static_cast<double>(lo);
    return (1.0 - w) * sorted_vals[lo] + w * sorted_vals[hi];
}

SummaryRow summarizeK(double k, double delta, const std::vector<SampleResult>& rows_for_k) {
    SummaryRow row;
    row.k = k;
    row.delta = delta;
    row.n_samples = static_cast<int>(rows_for_k.size());

    std::vector<double> escaped;
    escaped.reserve(rows_for_k.size());

    for (const auto& r : rows_for_k) {
        if (r.censored == 0) {
            escaped.push_back(r.t_esc_tau);
        }
    }

    row.n_escaped = static_cast<int>(escaped.size());
    row.n_censored = row.n_samples - row.n_escaped;

    if (!escaped.empty()) {
        std::sort(escaped.begin(), escaped.end());
        const double sum = std::accumulate(escaped.begin(), escaped.end(), 0.0);
        row.t_mean_escaped = sum / static_cast<double>(escaped.size());
        row.t_median_escaped = quantileSorted(escaped, 0.50);
        row.t_p25_escaped = quantileSorted(escaped, 0.25);
        row.t_p75_escaped = quantileSorted(escaped, 0.75);
    }

    return row;
}

void writeSamples(const std::string& out_path, const std::string& mode, double tau, double dt, double k_c, double t_max_tau, const std::vector<SampleResult>& rows) {
    const auto parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open samples output: " + out_path);
    }

    out << "mode\ttau\tdt\tk_c\tk\tdelta\tt_max_tau\tsample_id\tt_esc_tau\tcensored\n";
    out << std::fixed << std::setprecision(8);

    for (const auto& r : rows) {
        out << mode << "\t"
            << tau << "\t"
            << dt << "\t"
            << k_c << "\t"
            << r.k << "\t"
            << r.delta << "\t"
            << t_max_tau << "\t"
            << r.sample_id << "\t"
            << r.t_esc_tau << "\t"
            << r.censored << "\n";
    }
}

void writeSummary(const std::string& out_path, const std::string& mode, double tau, double dt, double k_c, double t_max_tau, const std::vector<SummaryRow>& rows) {
    const auto parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open summary output: " + out_path);
    }

    out << "mode\ttau\tdt\tk_c\tk\tdelta\tt_max_tau\tn_samples\tn_escaped\tn_censored\tt_mean_escaped\tt_median_escaped\tt_p25_escaped\tt_p75_escaped\n";
    out << std::fixed << std::setprecision(8);

    for (const auto& r : rows) {
        out << mode << "\t"
            << tau << "\t"
            << dt << "\t"
            << k_c << "\t"
            << r.k << "\t"
            << r.delta << "\t"
            << t_max_tau << "\t"
            << r.n_samples << "\t"
            << r.n_escaped << "\t"
            << r.n_censored << "\t"
            << r.t_mean_escaped << "\t"
            << r.t_median_escaped << "\t"
            << r.t_p25_escaped << "\t"
            << r.t_p75_escaped << "\n";
    }
}

void printUsage(const char* exe) {
    std::cerr
        << "Usage:\n  " << exe
        << " mode tau dt k_ic k_c t_warmup t_max_tau win_tau n_samples spacing_tau n_delta delta_min delta_max out_samples_tsv out_summary_tsv\n\n"
        << "mode: arrival | disappearance\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc != 16) {
            printUsage(argv[0]);
            return 1;
        }

        Config cfg;
        cfg.mode = toLower(argv[1]);
        cfg.tau = std::stod(argv[2]);
        cfg.dt = std::stod(argv[3]);
        cfg.k_ic = std::stod(argv[4]);
        cfg.k_c = std::stod(argv[5]);
        cfg.t_warmup = std::stod(argv[6]);
        cfg.t_max_tau = std::stod(argv[7]);
        cfg.win_tau = std::stod(argv[8]);
        cfg.n_samples = std::stoi(argv[9]);
        cfg.spacing_tau = std::stod(argv[10]);
        cfg.n_delta = std::stoi(argv[11]);
        cfg.delta_min = std::stod(argv[12]);
        cfg.delta_max = std::stod(argv[13]);
        cfg.out_samples = argv[14];
        cfg.out_summary = argv[15];

        if (cfg.mode != "arrival" && cfg.mode != "disappearance") {
            std::cerr << "Invalid mode: " << cfg.mode << " (expected arrival or disappearance)\n";
            return 1;
        }
        if (!(cfg.tau > 0.0 && cfg.dt > 0.0)) {
            std::cerr << "tau and dt must be positive\n";
            return 1;
        }
        if (!(cfg.delta_min > 0.0 && cfg.delta_max >= cfg.delta_min && cfg.n_delta >= 2)) {
            std::cerr << "Require: 0 < delta_min <= delta_max and n_delta >= 2\n";
            return 1;
        }
        if (cfg.n_samples < 1) {
            std::cerr << "n_samples must be >= 1\n";
            return 1;
        }

        std::cout << "Mode: " << cfg.mode << "\n";
        std::cout << "tau=" << cfg.tau << " dt=" << cfg.dt << "\n";
        std::cout << "k_ic=" << cfg.k_ic << " k_c=" << cfg.k_c << "\n";
        std::cout << "t_warmup=" << cfg.t_warmup << " t_max_tau=" << cfg.t_max_tau << " win_tau=" << cfg.win_tau << "\n";
        std::cout << "n_samples=" << cfg.n_samples << " spacing_tau=" << cfg.spacing_tau << "\n";
        std::cout << "delta in [" << cfg.delta_min << ", " << cfg.delta_max << "] with n=" << cfg.n_delta << "\n";

        // For tau=1 and dt=0.01 this is exactly 101 values in the delay buffer.
        const std::vector<std::vector<double>> ic_samples = collectICs(
            cfg.tau,
            cfg.dt,
            cfg.k_ic,
            cfg.t_warmup,
            cfg.n_samples,
            cfg.spacing_tau,
            2.0
        );

        const std::vector<double> deltas = geomspace(cfg.delta_min, cfg.delta_max, cfg.n_delta);

        std::vector<SampleResult> all_rows;
        all_rows.reserve(static_cast<size_t>(cfg.n_delta * cfg.n_samples));

        std::vector<SummaryRow> summaries;
        summaries.reserve(static_cast<size_t>(cfg.n_delta));

        for (int i = 0; i < cfg.n_delta; ++i) {
            const double delta = deltas[static_cast<size_t>(i)];
            const double k = (cfg.mode == "arrival") ? (cfg.k_c - delta) : (cfg.k_c + delta);

            std::vector<SampleResult> per_k;
            per_k.reserve(static_cast<size_t>(cfg.n_samples));

            for (int s = 0; s < cfg.n_samples; ++s) {
                const double t_esc_tau = escapeTimeTau(
                    cfg.tau,
                    cfg.dt,
                    k,
                    ic_samples[static_cast<size_t>(s)],
                    cfg.win_tau,
                    cfg.t_max_tau
                );

                SampleResult r;
                r.k = k;
                r.delta = delta;
                r.sample_id = s;
                r.t_esc_tau = t_esc_tau;
                r.censored = (t_esc_tau >= cfg.t_max_tau) ? 1 : 0;

                per_k.push_back(r);
                all_rows.push_back(r);
            }

            SummaryRow sr = summarizeK(k, delta, per_k);
            summaries.push_back(sr);

            std::cout << "  [" << (i + 1) << "/" << cfg.n_delta << "] k=" << std::fixed << std::setprecision(6)
                      << k << " escaped=" << sr.n_escaped << "/" << sr.n_samples
                      << " mean(Tesc|escaped)=" << sr.t_mean_escaped << " tau\n";
        }

        writeSamples(cfg.out_samples, cfg.mode, cfg.tau, cfg.dt, cfg.k_c, cfg.t_max_tau, all_rows);
        writeSummary(cfg.out_summary, cfg.mode, cfg.tau, cfg.dt, cfg.k_c, cfg.t_max_tau, summaries);

        std::cout << "Wrote samples: " << cfg.out_samples << "\n";
        std::cout << "Wrote summary: " << cfg.out_summary << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
