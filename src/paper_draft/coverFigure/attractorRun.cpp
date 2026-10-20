// Long deterministic trajectory of
//     dtheta/dt = -k sin(theta(t - tau)),   tau = 1
// for rendering the chaotic attractor on the cover figure.
//
// Same numerics as the paper: explicit trapezoidal rule on a grid with tau/dt
// an integer, so the delayed values are exact ring-buffer lookups.  The initial
// history is constant, theta(s) = theta0 for s in [-tau, 0].
//
// Output is raw float64 theta, one value every record_dt, after the warm-up.
//
// usage: ./attractorRun tauk theta0 dt t_warmup t_measure record_dt out_bin

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 8)
    {
        std::fprintf(stderr, "usage: %s tauk theta0 dt t_warmup t_measure record_dt out_bin\n", argv[0]);
        return 1;
    }
    const double tau = 1.0;
    const double k = std::atof(argv[1]) / tau;
    const double theta0 = std::atof(argv[2]);
    const double dt = std::atof(argv[3]);
    const double t_warmup = std::atof(argv[4]);
    const double t_measure = std::atof(argv[5]);
    const double record_dt = std::atof(argv[6]);

    const long n_delay = std::lround(tau / dt);
    const long n_warmup = std::lround(t_warmup / dt);
    const long n_measure = std::lround(t_measure / dt);
    const long stride = std::lround(record_dt / dt);

    // theta_j lives in slot j mod (n_delay + 1); before step n the slot (n + 1)
    // holds theta_{n - n_delay} and slot (n + 2) holds theta_{n + 1 - n_delay}.
    std::vector<double> hist(n_delay + 1, theta0);
    double theta = theta0;

    FILE *f = std::fopen(argv[7], "wb");
    std::vector<double> chunk;
    chunk.reserve(1 << 20);

    for (long n = 0; n < n_warmup + n_measure; ++n)
    {
        const double delayed_now = hist[(n + 1) % (n_delay + 1)];
        const double delayed_next = hist[(n + 2) % (n_delay + 1)];
        theta += -0.5 * dt * k * (std::sin(delayed_now) + std::sin(delayed_next));
        hist[(n + 1) % (n_delay + 1)] = theta;

        if (n >= n_warmup && (n - n_warmup + 1) % stride == 0)
        {
            chunk.push_back(theta);
            if (chunk.size() == chunk.capacity())
            {
                std::fwrite(chunk.data(), sizeof(double), chunk.size(), f);
                chunk.clear();
            }
        }
    }
    std::fwrite(chunk.data(), sizeof(double), chunk.size(), f);
    std::fclose(f);
    return 0;
}
