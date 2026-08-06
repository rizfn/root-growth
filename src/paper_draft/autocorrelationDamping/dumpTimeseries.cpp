// Same integrator as phasediff.cpp, but dumps theta timeseries (float32) so the
// autocorrelation function itself can be measured directly.
// usage: ./dumpts k eta tmax nreal seed outfile [dt] [record_dt]
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
    double dt        = (argc > 7) ? std::atof(argv[7]) : 1e-3;
    double record_dt = (argc > 8) ? std::atof(argv[8]) : 0.1;

    const double tau = 1.0, theta0 = 1.5708;
    const int  ndelay = (int)std::llround(tau / dt);
    const long nsteps = (long)std::llround(tmax / dt);
    const int  rec_every = (int)std::llround(record_dt / dt);
    const double sqdt = std::sqrt(dt);

    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    FILE *f = fopen(outfile.c_str(), "wb");
    int nrec = (int)(nsteps / rec_every);
    fwrite(&nreal, sizeof(int), 1, f);
    fwrite(&nrec,  sizeof(int), 1, f);
    fwrite(&record_dt, sizeof(double), 1, f);

    std::vector<double> hist(ndelay + 1);
    std::vector<float> buf(nrec);

    for (int r = 0; r < nreal; ++r)
    {
        for (int i = 0; i <= ndelay; ++i) hist[i] = theta0;
        int idx = 0;
        double theta = theta0;
        long ir = 0;
        for (long s = 0; s < nsteps; ++s)
        {
            int i_del  = (idx + 1) % (ndelay + 1);
            int i_del2 = (idx + 2) % (ndelay + 1);
            double k1 = -k * std::sin(hist[i_del]);
            double k2 = -k * std::sin(hist[i_del2]);
            theta += 0.5 * (k1 + k2) * dt + eta * nd(gen) * sqdt;
            idx = i_del;
            hist[idx] = theta;
            if ((s + 1) % rec_every == 0 && ir < nrec) buf[ir++] = (float)theta;
        }
        fwrite(buf.data(), sizeof(float), nrec, f);
    }
    fclose(f);
    return 0;
}
