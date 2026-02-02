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

// Undamped harmonic oscillator: d²θ/dt² + ω₀²·θ = η·ξ(t)
// In first-order form: dθ/dt = ω,  dω/dt = -ω₀²·θ + η·ξ(t)
// Integrated using SYMPLECTIC EULER for energy conservation
constexpr double DEFAULT_K = 0.5;              // Natural frequency ω₀ (rad/s)
constexpr double DEFAULT_GAMMA = 0.0;          // Damping coefficient (0 = undamped)
constexpr double DEFAULT_ETA = 0.1;            // Noise strength
constexpr double DEFAULT_THETA0 = 1.5708;      // Initial angle (π/2 radians = 90 degrees)
constexpr double DEFAULT_OMEGA0 = 0.0;         // Initial angular velocity
constexpr double DEFAULT_DT = 0.1;             // Time step
constexpr double DEFAULT_T_MAX = 1000.0;       // Total simulation time

// Solve using symplectic Euler method (energy-conserving for harmonic oscillators)
void solveOscillator(std::ofstream &file, double k, double gamma, double eta, double theta0, double omega0, double dt, double t_max)
{
    double t = 0.0;
    double theta = theta0;
    double omega = omega0;
    
    // Write header
    file << "time\ttheta\tomega\n";
    
    // Write initial condition
    file << std::fixed << std::setprecision(6) << t << "\t" << theta << "\t" << omega << "\n";
    
    int n_steps = static_cast<int>(t_max / dt);
    double k_squared = k * k;
    
    for (int step = 0; step < n_steps; ++step)
    {
        // Symplectic Euler method for second-order stochastic ODE
        // This preserves energy and prevents numerical drift in oscillators
        // Order: update position first, then velocity using NEW position
        // dθ = ω·dt
        // dω = (-ω₀²·θ_new - γ·ω)·dt + η·dW
        
        double noise = eta * normal_dist(gen) * std::sqrt(dt);
        
        // Symplectic Euler: position-then-velocity update
        theta = theta + omega * dt;  // Update position with old velocity
        omega = omega + (-k_squared * theta - gamma * omega) * dt + noise;  // Update velocity with NEW position
        t = t + dt;
        
        // Write to file
        file << t << "\t" << theta << "\t" << omega << "\n";
        
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
    double gamma = DEFAULT_GAMMA;
    double eta = DEFAULT_ETA;
    double theta0 = DEFAULT_THETA0;
    double omega0 = DEFAULT_OMEGA0;
    double dt = DEFAULT_DT;
    double t_max = DEFAULT_T_MAX;
    int sim_no = 0;

    if (argc > 1)
        k = std::stod(argv[1]);
    if (argc > 2)
        gamma = std::stod(argv[2]);
    if (argc > 3)
        eta = std::stod(argv[3]);
    if (argc > 4)
        theta0 = std::stod(argv[4]);
    if (argc > 5)
        omega0 = std::stod(argv[5]);
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
    folderStream << exeDir << "/outputs/SimpleOscillator/k_" << k
                 << "_gamma_" << gamma
                 << "_theta0_" << theta0 << "_omega0_" << omega0
                 << "_dt_" << dt << "_tmax_" << t_max;
    std::string folderPath = folderStream.str();
    
    // Create file path within folder
    std::ostringstream filePathStream;
    filePathStream << folderPath << "/eta_" << eta << "_simNo_" << sim_no << ".tsv";
    std::string filePath = filePathStream.str();

    // Create directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filePath).parent_path());

    std::ofstream file;
    file.open(filePath);

    solveOscillator(file, k, gamma, eta, theta0, omega0, dt, t_max);

    file.close();

    std::cout << "Symplectic oscillator solution complete. Results saved to: " << filePath << std::endl;

    return 0;
}