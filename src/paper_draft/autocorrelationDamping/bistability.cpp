// Coexistence of the two delay-induced cycles of
//     dtheta/dt = -k sin(theta(t-tau)) + eta xi(t),   tau = 1
//
// Mode "cont":  deterministic continuation in k.  Starting from a converged
//               cycle, k is stepped up (then down) and the orbit re-relaxed at
//               each step, carrying the history across.  A hysteresis loop in
//               the branch amplitude reveals the bistable window.
// Mode "ic":    relax deterministically from a sinusoidal initial history
//               theta(s) = A sin(2 pi s / P), s in [-tau, 0], to see which
//               basin that initial condition lies in.
// Mode "esc":   noisy escape in a chosen direction, from a converged cycle;
//               prints one first-passage time per realisation (-1 = no escape).
//
// usage:
//   ./bistability cont  <k_lo> <k_hi> <dk> <A0> <P0>
//   ./bistability ic    <k> <A> <P>
//   ./bistability esc   <k> <eta> <A0> <P0> <tmax> <nreal> <seed>

#include <vector>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <cstdlib>

static const double TAU = 1.0;
static const double DT = 1e-3;
static const int NDELAY = (int)(TAU / DT + 0.5);

struct Orbit { std::vector<double> hist; int idx; double theta; };

static Orbit seed_orbit(double A, double P)
{
    Orbit S;
    S.hist.resize(NDELAY + 1);
    // hist[(idx+1+j) % (ndelay+1)] holds theta at time -(tau) + j*dt
    for (int j = 0; j <= NDELAY; ++j)
    {
        double s = -TAU + j * DT;
        S.hist[j] = A * std::sin(2.0 * M_PI * s / P);
    }
    S.idx = NDELAY;
    S.theta = S.hist[NDELAY];
    return S;
}

static inline void step(Orbit &S, double k, double noise)
{
    int i1 = (S.idx + 1) % (NDELAY + 1);
    int i2 = (S.idx + 2) % (NDELAY + 1);
    double a = -k * std::sin(S.hist[i1]);
    double b = -k * std::sin(S.hist[i2]);
    S.theta += 0.5 * (a + b) * DT + noise;
    S.idx = i1;
    S.hist[S.idx] = S.theta;
}

// relax deterministically, then report peak amplitude and period
static void characterise(Orbit &S, double k, double t_relax, double t_meas,
                         double &amp, double &period)
{
    long nrel = (long)(t_relax / DT + 0.5);
    for (long s = 0; s < nrel; ++s) step(S, k, 0.0);

    long nmeas = (long)(t_meas / DT + 0.5);
    amp = 0.0;
    std::vector<double> cross;
    double prev = S.theta, t = 0.0;
    bool armed = false;
    double h = 0.0;
    // first pass to fix the hysteresis threshold
    Orbit probe = S;
    double pmax = 0.0;
    for (long s = 0; s < nmeas / 4; ++s) { step(probe, k, 0.0); pmax = std::max(pmax, std::fabs(probe.theta)); }
    h = 0.4 * pmax;

    for (long s = 0; s < nmeas; ++s)
    {
        prev = S.theta;
        step(S, k, 0.0);
        t += DT;
        amp = std::max(amp, std::fabs(S.theta));
        if (S.theta < -h) armed = true;
        if (armed && prev <= 0.0 && S.theta > 0.0)
        {
            cross.push_back(t - DT + DT * (-prev / (S.theta - prev)));
            armed = false;
        }
    }
    period = (cross.size() >= 2)
             ? (cross.back() - cross.front()) / (double)(cross.size() - 1)
             : NAN;
}

int main(int argc, char **argv)
{
    std::string mode = argv[1];

    if (mode == "cont")
    {
        double klo = std::atof(argv[2]), khi = std::atof(argv[3]), dk = std::atof(argv[4]);
        double A0 = std::atof(argv[5]), P0 = std::atof(argv[6]);
        Orbit S = seed_orbit(A0, P0);
        double amp, per;
        characterise(S, klo, 4000.0, 200.0, amp, per);
        printf("# dir\tk\tamp\tperiod\n");
        for (double k = klo; k <= khi + 1e-9; k += dk)
        {
            characterise(S, k, 600.0, 200.0, amp, per);
            printf("up\t%.4f\t%.6f\t%.6f\n", k, amp, per);
        }
        for (double k = khi; k >= klo - 1e-9; k -= dk)
        {
            characterise(S, k, 600.0, 200.0, amp, per);
            printf("down\t%.4f\t%.6f\t%.6f\n", k, amp, per);
        }
        return 0;
    }

    // Trace one branch: seed it at k_seed, then walk k down to k_lo and up to
    // k_hi, re-relaxing at each step.  The branch ends where the orbit jumps to
    // the other cycle, which locates the fold.
    if (mode == "branch")
    {
        double kseed = std::atof(argv[2]);
        double A0 = std::atof(argv[3]), P0 = std::atof(argv[4]);
        double klo = std::atof(argv[5]), khi = std::atof(argv[6]), dk = std::atof(argv[7]);
        double amp, per;
        printf("# dir\tk\tamp\tperiod\n");
        for (int dir = 0; dir < 2; ++dir)
        {
            Orbit S = seed_orbit(A0, P0);
            characterise(S, kseed, 4000.0, 200.0, amp, per);
            double kend = dir == 0 ? klo : khi;
            double step_k = dir == 0 ? -dk : dk;
            for (double k = kseed; (dir == 0 ? k >= kend - 1e-9 : k <= kend + 1e-9); k += step_k)
            {
                characterise(S, k, 800.0, 200.0, amp, per);
                printf("%s\t%.4f\t%.6f\t%.6f\n", dir == 0 ? "down" : "up", k, amp, per);
            }
        }
        return 0;
    }

    // Dump the deterministic orbit of one branch, sampled every tau, so the two
    // coexisting cycles can be drawn as return maps.
    if (mode == "dump")
    {
        double k = std::atof(argv[2]), A = std::atof(argv[3]), P = std::atof(argv[4]);
        int npts = std::atoi(argv[5]);
        Orbit S = seed_orbit(A, P);
        double amp, per;
        characterise(S, k, 4000.0, 200.0, amp, per);
        fprintf(stderr, "# amp=%.6f period=%.6f\n", amp, per);
        long per_tau = (long)(TAU / DT + 0.5);
        for (int i = 0; i < npts; ++i)
        {
            for (long s = 0; s < per_tau; ++s) step(S, k, 0.0);
            printf("%.8f\n", S.theta);
        }
        return 0;
    }

    // Dump theta(t) finely along one branch, so the return map can be drawn as
    // the continuous closed curve (theta(t), theta(t+tau)) rather than as
    // stroboscopic dots.
    if (mode == "trace")
    {
        double k = std::atof(argv[2]), A = std::atof(argv[3]), P = std::atof(argv[4]);
        double tspan = std::atof(argv[5]);
        double rec = (argc > 6) ? std::atof(argv[6]) : 0.005;
        Orbit S = seed_orbit(A, P);
        double amp, per;
        characterise(S, k, 4000.0, 200.0, amp, per);
        fprintf(stderr, "# amp=%.6f period=%.6f\n", amp, per);
        long nsteps = (long)(tspan / DT + 0.5);
        int every = (int)(rec / DT + 0.5);
        for (long s = 0; s < nsteps; ++s)
        {
            step(S, k, 0.0);
            if ((s + 1) % every == 0) printf("%.8f\n", S.theta);
        }
        return 0;
    }

    if (mode == "ic")
    {
        double k = std::atof(argv[2]), A = std::atof(argv[3]), P = std::atof(argv[4]);
        Orbit S = seed_orbit(A, P);
        double amp, per;
        characterise(S, k, 4000.0, 400.0, amp, per);
        printf("%.4f\t%.4f\t%.4f\t%.6f\t%.6f\n", k, A, P, amp, per);
        return 0;
    }

    if (mode == "esc")
    {
        double k = std::atof(argv[2]), eta = std::atof(argv[3]);
        double A0 = std::atof(argv[4]), P0 = std::atof(argv[5]);
        double tmax = std::atof(argv[6]);
        int nreal = std::atoi(argv[7]);
        unsigned seed = (unsigned)std::atoi(argv[8]);

        // converge the noiseless target cycle once
        Orbit base = seed_orbit(A0, P0);
        double amp0, per0;
        characterise(base, k, 4000.0, 200.0, amp0, per0);
        const bool start_is_long = (per0 > 4.8);
        fprintf(stderr, "# start amp=%.4f period=%.4f (%s branch)\n",
                amp0, per0, start_is_long ? "long" : "period-4tau");

        // Discriminate by the number of hysteretic upward crossings in a window:
        // the period-4tau cycle gives W/4.01 of them, the long cycle W/5.52.
        // Amplitude is NOT usable -- at large eta the noise alone pushes peaks
        // of the small cycle past the amplitude of the large one.
        const double window = 200.0;
        const long wsteps = (long)(window / DT + 0.5);
        const double n_p4 = window / 4.007, n_long = window / 5.521;
        const double n_thresh = 0.5 * (n_p4 + n_long);
        const double h = 0.4 * amp0;

        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        const double sqdt = std::sqrt(DT);
        const long nsteps = (long)(tmax / DT + 0.5);

        for (int r = 0; r < nreal; ++r)
        {
            Orbit S = base;
            double t = 0.0, tesc = -1.0, prev = S.theta;
            long wcount = 0, ncross = 0;
            bool armed = false;
            for (long s = 0; s < nsteps; ++s)
            {
                prev = S.theta;
                step(S, k, eta * nd(gen) * sqdt);
                t += DT;
                if (S.theta < -h) armed = true;
                if (armed && prev <= 0.0 && S.theta > 0.0) { ++ncross; armed = false; }
                if (++wcount >= wsteps)
                {
                    bool now_long = ((double)ncross < n_thresh);
                    if (now_long != start_is_long) { tesc = t; break; }
                    wcount = 0; ncross = 0;
                }
            }
            printf("%.3f\n", tesc);
        }
        return 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
