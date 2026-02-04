#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <random>

#pragma GCC optimize("inline", "unroll-loops", "no-stack-protector")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native", "f16c")

static auto _ = []()
{std::ios_base::sync_with_stdio(false);std::cin.tie(nullptr);std::cout.tie(nullptr);return 0; }();

std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<double> normal_dist(0.0, 1.0);

// Oscillator with noise in the phase: d²θ/dt² = -ω₀²·sin(θ + ξ), where dξ = η·dW
// In first-order form: 
//   dθ/dt = ω
//   dω/dt = -ω₀²·sin(θ + ξ)
//   dξ/dt = η·ξ(t)  (ξ accumulates as Brownian motion)
// Integrated using SYMPLECTIC-like method
constexpr double DEFAULT_K = 0.5;              // Natural frequency ω₀ (rad/s)
constexpr double DEFAULT_ETA = 0.1;            // Phase noise strength
constexpr double DEFAULT_THETA0 = 1.5708;      // Initial angle (π/2 radians = 90 degrees)
constexpr double DEFAULT_OMEGA0 = 0.0;         // Initial angular velocity
constexpr double DEFAULT_XI0 = 0.0;            // Initial accumulated phase noise
constexpr double DEFAULT_DT = 0.1;             // Time step
constexpr double DEFAULT_T_MAX = 1000.0;       // Total simulation time

// Solve using symplectic-like method
void solveOscillatorPhaseNoise(std::ofstream &file, double k, double eta, double theta0, double omega0, double xi0, double dt, double t_max)
{
    double t = 0.0;
    double theta = theta0;
    double omega = omega0;
    double xi = xi0;  // Accumulated phase noise
    
    // Write header
    file << "time\ttheta\tomega\txi\n";
    
    // Write initial condition
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\t" << omega << "\t" << xi << "\n";
    
    int n_steps = static_cast<int>(t_max / dt);
    double k_squared = k * k;
    
    for (int step = 0; step < n_steps; ++step)
    {
        // Generate noise for this time step
        double noise = eta * normal_dist(gen) * std::sqrt(dt);
        
        // Update accumulated phase noise (Brownian motion)
        xi = xi + noise;
        
        // Symplectic-like Euler: position-then-velocity update
        // dθ = ω·dt
        // dω = -ω₀²·sin(θ_new + ξ)·dt
        theta = theta + omega * dt;  // Update position with old velocity
        omega = omega - k_squared * std::sin(theta + xi) * dt;  // Update velocity with NEW position and current phase noise
        t = t + dt;
        
        // Write to file
        file << t << "\t" << theta << "\t" << omega << "\t" << xi << "\n";
        
        // Print progress every 10%
        if (step % (n_steps / 10) == 0)
        {
            double progress = 100.0 * step / n_steps;
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
        }
    }
    
    std::cout << "\rProgress: 100.0%  " << std::endl;
}

int main(int argc, char *argv[])
{
    double k = DEFAULT_K;
    double eta = DEFAULT_ETA;
    double theta0 = DEFAULT_THETA0;
    double omega0 = DEFAULT_OMEGA0;
    double xi0 = DEFAULT_XI0;
    double dt = DEFAULT_DT;
    double t_max = DEFAULT_T_MAX;
    int sim_no = 0;

    if (argc > 1)
        k = std::stod(argv[1]);
    if (argc > 2)
        eta = std::stod(argv[2]);
    if (argc > 3)
        theta0 = std::stod(argv[3]);
    if (argc > 4)
        omega0 = std::stod(argv[4]);
    if (argc > 5)
        xi0 = std::stod(argv[5]);
    if (argc > 6)
        dt = std::stod(argv[6]);
    if (argc > 7)
        t_max = std::stod(argv[7]);
    if (argc > 8)
        sim_no = std::stoi(argv[8]);

    std::string exePath = argv[0];
    std::string exeDir = std::filesystem::path(exePath).parent_path().string();
    
    // Create folder for this parameter set
    std::ostringstream folderStream;
    folderStream << exeDir << "/outputs/PhaseNoiseOscillator/k_" << k
                 << "_theta0_" << theta0 << "_omega0_" << omega0
                 << "_xi0_" << xi0 << "_dt_" << dt << "_tmax_" << t_max;
    std::string folderPath = folderStream.str();
    
    // Create file path within folder
    std::ostringstream filePathStream;
    filePathStream << folderPath << "/eta_" << eta << "_simNo_" << sim_no << ".tsv";
    std::string filePath = filePathStream.str();

    // Create directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);

    solveOscillatorPhaseNoise(file, k, eta, theta0, omega0, xi0, dt, t_max);

    file.close();

    std::cout << "Phase noise oscillator solution complete. Results saved to: " << filePath << std::endl;

    return 0;
}
