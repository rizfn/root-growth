// Infinitesimal phase-response curve Z(phi) of the deterministic limit cycle of
//     dtheta/dt = -k sin(theta(t-tau))
// measured by direct impulse perturbation, and the resulting phase-diffusion
// coefficient for additive noise:
//     D = (eta^2 / 2) * (1/T) * int_0^T Z(phi)^2 dphi     (per unit eta^2)
//
// usage: ./prc k [nphase] [dt]
// prints:  k  T  c_theory   (c_theory = D/eta^2)
// and, on stderr, the PRC itself as "phase  Z".

#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>

struct State { std::vector<double> hist; int idx; double theta; };

int main(int argc, char **argv)
{
    double k      = std::atof(argv[1]);
    int    nphase = (argc > 2) ? std::atoi(argv[2]) : 200;
    double dt     = (argc > 3) ? std::atof(argv[3]) : 1e-4;

    const double tau = 1.0, theta0 = 1.5708;
    const int ndelay = (int)std::llround(tau / dt);

    auto step = [&](State &S) {
        int i1 = (S.idx + 1) % (ndelay + 1);
        int i2 = (S.idx + 2) % (ndelay + 1);
        double a = -k * std::sin(S.hist[i1]);
        double b = -k * std::sin(S.hist[i2]);
        S.theta += 0.5 * (a + b) * dt;
        S.idx = i1;
        S.hist[S.idx] = S.theta;
    };

    // ---- 1. relax onto the limit cycle -------------------------------------
    State S0{std::vector<double>(ndelay + 1, theta0), 0, theta0};
    long nrelax = (long)std::llround(3000.0 / dt);
    for (long s = 0; s < nrelax; ++s) step(S0);

    // ---- 2. find the period from two successive upward zero crossings ------
    auto next_crossing = [&](State S, double &tcross, long maxsteps) -> bool {
        double t = 0.0;
        for (long s = 0; s < maxsteps; ++s) {
            double prev = S.theta;
            step(S);
            t += dt;
            if (prev <= 0.0 && S.theta > 0.0) { tcross = t - dt * (S.theta / (S.theta - prev)); return true; }
        }
        return false;
    };
    // advance to the first crossing so phase 0 is well defined
    {
        double tc;
        long maxs = (long)std::llround(50.0 / dt);
        double t = 0.0;
        for (long s = 0; s < maxs; ++s) {
            double prev = S0.theta; step(S0); t += dt;
            if (prev <= 0.0 && S0.theta > 0.0) break;
        }
        (void)tc;
    }
    double T;
    {
        State S = S0; double tc;
        next_crossing(S, tc, (long)std::llround(50.0 / dt));
        T = tc;
    }

    // ---- 3. impulse response ----------------------------------------------
    // Perturb by +delta at phase phi, integrate long enough for the transient to
    // die, and read off the asymptotic timing shift relative to the unperturbed
    // orbit.  Z = -omega * dt_shift / delta.
    const double delta = 1e-5;
    const double omega = 2.0 * M_PI / T;
    const long nsettle = (long)std::llround(400.0 / dt);
    const int  ncycles_ref = 100;

    std::vector<double> Z(nphase);
    long steps_per_phase = (long)std::llround(T / dt) / nphase;

    State Sp = S0;   // walks around the cycle, phase 0 at a crossing
    for (int ip = 0; ip < nphase; ++ip)
    {
        // reference: crossing time after nsettle steps, no perturbation
        auto measure = [&](State S) {
            for (long s = 0; s < nsettle; ++s) step(S);
            double tc; next_crossing(S, tc, (long)std::llround(50.0 / dt));
            return tc;
        };
        double t_ref = measure(Sp);
        State Sq = Sp;
        Sq.theta += delta;               // impulse: theta jumps, history unchanged
        Sq.hist[Sq.idx] = Sq.theta;
        double t_per = measure(Sq);
        double shift = t_per - t_ref;
        // unwrap onto (-T/2, T/2]
        while (shift >  T / 2) shift -= T;
        while (shift <= -T / 2) shift += T;
        Z[ip] = -omega * shift / delta;
        fprintf(stderr, "%g\t%g\n", ip * T / nphase, Z[ip]);

        for (long s = 0; s < steps_per_phase; ++s) step(Sp);
    }
    (void)ncycles_ref;

    double meansq = 0.0;
    for (double z : Z) meansq += z * z;
    meansq /= nphase;
    printf("%g\t%.8f\t%.8f\n", k, T, 0.5 * meansq);
    return 0;
}
