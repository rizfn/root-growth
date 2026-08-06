# Re-analysis of the autocorrelation damping ($\tau = 1$)

Re-examines the claim that the damping rate $\lambda$ of $C(\Delta)=\langle\theta(t)\theta(t-\Delta)\rangle$
is constant at low noise and only starts scaling with $\eta$ above a threshold.

## What was wrong

The published damping rates came from fitting $A e^{-\lambda\Delta}\cos(\omega\Delta+\varphi)$
to $C(\Delta)$ over $\Delta \le 240$, from runs of $t_{\max}=1000$.

Running that **same fit on noiseless ($\eta = 0$) data** returns

| $k$ | $\lambda_{\rm fit}(\eta=0)$ |
|-----|------------------------------|
| 2   | $2.0\times10^{-15}$ |
| 4   | $4.69\times10^{-6}$ |

At $k=4$ that is exactly the height of the published plateau, and the plateau
ends precisely where the true $\lambda = 0.65\,\eta^2$ crosses it
($\eta \approx 2.7\times10^{-3}$). The plateau is the fit's own floor: over a
$\Delta\le240$ window an envelope decay of $\lambda \lesssim 10^{-5}$ is
smaller than the anharmonicity of the $k=4$ waveform, which is not a single
cosine, so the fit reports the mismatch as "damping".

The integrator itself is fine: with a fixed-step ring buffer the measured phase
diffusion at $\eta=0$ is $\sim10^{-25}$ (i.e. zero to machine precision), and
$D$ is unchanged over $dt \in [2.5\times10^{-4}, 2\times10^{-3}]$.

## What is actually true

`phaseDiffusion.cpp` measures the phase-diffusion coefficient $D$ directly from
the ensemble variance of the $n$-th hysteretic upward zero crossing,
$\mathrm{Var}(t_n) = 2D\,t_n/\omega^2$. For $\theta\simeq A\cos\phi$ with
diffusing phase this is exactly the autocorrelation damping, $\lambda = D$
(verified independently against the envelope of $C(\Delta)$ from
$t_{\max}=6\times10^{4}$ runs).

On a **single attractor** (the period-$4\tau$ branch, $\pi/2 < k \lesssim 3.85$)
there is **no transition**: $\lambda = c(k)\,\eta^2$ holds over four decades of
$\eta$, and $c(k)$ agrees with standard phase reduction,
$c(k) = \tfrac12\langle Z^2\rangle$, to 1–3% (`prc.cpp`).

One caveat on the estimator. Close to the Hopf point the cycle amplitude is
small, so above some $\eta$ the noise adds spurious zero crossings and the
cycle counter over-reports $D$. The onset scales with the amplitude: at
$k\tau=1.6$ ($A=0.38$) it sets in by $\eta \approx 0.015$, at $k\tau=1.8$
($A=0.62$) not until $\eta \approx 0.15$. The figures use $k\tau = 1.8, 2, 3$
and cap $k\tau=1.8$ at $\eta \le 0.1$, where $\gamma/\eta^2 = 0.36$–$0.43$
against a predicted $0.389$. Where the counter fails, the $C(\Delta)$ envelope
is unaffected and remains the reliable measure.

A **real** sharp transition exists where two cycles coexist, e.g. $k=4$, where
the $T=5.52$ cycle and the period-$4\tau$ cycle are both stable. Noise drives an
escape from the former to the latter with a Kramers rate

$$r = r_0\,e^{-C/\eta^{2}},\qquad C = 8.84\times10^{-4},\quad r_0 = 0.027$$

straight over 3.1 decades of rate (residual s.d. 0.11 in $\ln r$). The fit uses
only $\eta \le 0.015$: beyond that the mean escape time approaches the
200-time-unit detection window and the measured rate saturates at $\sim1/200$.

Because $r$ depends on $\eta$ exponentially, a finite observation window makes
this look like a sharp threshold at $\eta_c = \sqrt{C/\ln(T_{\rm obs} r_0)}$ —
only logarithmically dependent on $T_{\rm obs}$: $\eta_c = 0.017$ for
$T_{\rm obs}=800$, $0.013$ for $10^4$, $0.010$ for $3\times10^5$. A factor 300
in observation time moves the apparent threshold by 40%.

## Structure of the bistability (`bistability.cpp`)

Continuation in $k$ shows the two cycles coexist only over

$$k\tau \in [3.985,\ 4.240]$$

The period-$4\tau$ branch runs continuously from the Hopf point up to
$k\tau\approx5.0$ (beyond which $\theta$ runs away); the long cycle exists only
inside that narrow window, folding back onto the period-$4\tau$ branch at both
ends. $k\tau=4$, used in the paper, sits inside it.

The escape is **one-way**. Starting on the long cycle, escape to the
period-$4\tau$ cycle is routine; starting on the period-$4\tau$ cycle, escape to
the long cycle was never observed — 0/24 realisations at each of
$\eta = 0.01, 0.02, 0.03, 0.05, 0.10$, over $2\times10^{5}$ time units each,
bounding the reverse rate below $2\times10^{-7}$ against a forward rate of
$5\times10^{-3}$ at $\eta = 0.02$. Independently, in long noisy runs the
fraction of time spent at $T>4.8$ is exactly zero for $\eta \ge 0.05$.
(An amplitude-based discriminator gives false positives here: at large $\eta$
noise pushes peaks of the small cycle past the amplitude of the large one.
Discriminate by period / cycle count instead.)

So the long cycle is metastable under noise and the period-$4\tau$ cycle is the
robust state. Deterministically both are genuine attractors, and which one is
reached depends on the initial history — a constant history $\theta\equiv\pi/2$
lands on the long cycle, whereas sinusoidal histories of amplitude $\lesssim3$
land on the period-$4\tau$ cycle.

## Files

| file | purpose |
|------|---------|
| `phaseDiffusion.cpp` | ring-buffer SDDE solver; writes hysteretic zero-crossing times |
| `dumpTimeseries.cpp` | same solver; dumps $\theta(t)$ for autocorrelations |
| `prc.cpp` | infinitesimal phase-response curve $Z(\phi)$ by impulse perturbation, and $\tfrac12\langle Z^2\rangle$ |
| `bistability.cpp` | continuation in $k$, basin tests, both-direction escape, orbit traces |
| `readCrossings.py` | reader + the $\mathrm{Var}(t_n)$ estimator for $D$ |
| `vizPaperFigs.py` | the paper figures, into `plots/paper/` |

Build with `g++ -O3 -march=native -std=c++17 <file>.cpp -o <file>`.

The figures read one reduced file, `outputs/damping_analysis.npz`, which holds
the measured $D(k,\eta)$, the PRC scan, the displayed $C(\Delta)$ curves, the
switching statistics and the Kramers fit. `outputs/` is gitignored, so the npz
is not committed; regenerate it by rerunning the solvers above.
