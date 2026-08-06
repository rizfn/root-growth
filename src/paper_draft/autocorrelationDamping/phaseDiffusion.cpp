// v2: same integrator, but the phase marker is a HYSTERETIC upward zero
// crossing.  A crossing is only counted once theta has first gone below -h,
// which removes the spurious re-crossings that noise produces right at theta=0
// and which otherwise corrupt the cycle count at large eta.
//
// h is set to 0.4 * RMS(theta) measured during the transient.
//
// usage: ./phasediff2 k eta tmax nreal seed outfile [dt]

#include <vector>
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include <cstdlib>

int main(int argc, char **argv)
{
    double k     = std::atof(argv[1]);
    double eta   = std::atof(argv[2]);
    double tmax  = std::atof(argv[3]);
    int    nreal = std::atoi(argv[4]);
    unsigned seed = (unsigned)std::atoi(argv[5]);
    std::string outfile = argv[6];
    double dt    = (argc > 7) ? std::atof(argv[7]) : 1e-3;

    const double tau = 1.0, theta0 = 1.5708;
    const int  ndelay = (int)std::llround(tau / dt);
    const long nsteps = (long)std::llround(tmax / dt);
    const double sqdt = std::sqrt(dt);
    const double t_transient = 200.0;
    const long  s_transient = (long)std::llround(t_transient / dt);

    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    FILE *f = fopen(outfile.c_str(), "wb");
    fwrite(&nreal, sizeof(int), 1, f);

    std::vector<double> hist(ndelay + 1);
    std::vector<double> cross;
    cross.reserve(1 << 16);

    for (int r = 0; r < nreal; ++r)
    {
        for (int i = 0; i <= ndelay; ++i) hist[i] = theta0;
        int idx = 0;
        double theta = theta0, prev = theta0, t = 0.0;
        double sumsq = 0.0; long nsq = 0;
        double h = 0.0;
        bool armed = false;
        cross.clear();

        for (long s = 0; s < nsteps; ++s)
        {
            int i_del  = (idx + 1) % (ndelay + 1);
            int i_del2 = (idx + 2) % (ndelay + 1);
            double k1 = -k * std::sin(hist[i_del]);
            double k2 = -k * std::sin(hist[i_del2]);
            prev = theta;
            theta += 0.5 * (k1 + k2) * dt + eta * nd(gen) * sqdt;
            t += dt;
            idx = i_del;
            hist[idx] = theta;

            if (s < s_transient) { if (s > s_transient / 2) { sumsq += theta * theta; ++nsq; } continue; }

            if (s == s_transient) h = 0.4 * std::sqrt(sumsq / (double)nsq);

            if (theta < -h) armed = true;
            if (armed && prev <= 0.0 && theta > 0.0)
            {
                double frac = -prev / (theta - prev);
                cross.push_back(t - dt + frac * dt);
                armed = false;
            }
        }

        long nc = (long)cross.size();
        fwrite(&nc, sizeof(long), 1, f);
        fwrite(cross.data(), sizeof(double), nc, f);
    }
    fclose(f);
    return 0;
}
