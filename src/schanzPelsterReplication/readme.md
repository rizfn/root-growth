## Replicating Schanz and Pelster's Results

In their [PRE paper from 2003](https://doi.org/10.1103/PhysRevE.67.056205), Schanz and Pelster study the deterministic system

$$ \frac{d\theta}{dt} = -k \,\sin(\theta(t-\tau)) $$

In my opinion, it's the most comprehensive study of it, including more details than more recent papers such as [Sprott's in Physics Letters A, 2007](https://doi.org/10.1016/j.physleta.2007.01.083).

There are some random checks in this folder to study their work, as it's the main thing I could find which discusses the second limit cycle.

### Amplitude Scaling

After weeks, I've finally figured out that the second limit cycle forms only due to the amplitude becoming greater than $\pi$, as that causes the sine function to swap signs.

Now, I want to derive an approximation of how the amplitude scales with $k\tau$.

The solution is **not** sinusoidal, but let's assume one:

$$\theta(t) = R\cos(\omega t)$$

Then, 

$$
\begin{align*}
\frac{d\theta}{dt} &= -R\omega \sin(\omega t) \\
\theta(t-\tau) &= R\cos(\omega t - \omega\tau)
\end{align*}
$$

Our DDE is 

$$ \frac{d\theta}{dt} = -k \,\sin(\theta(t-\tau)) $$

We need to consider $\sin(R\cos(\phi))$, as that appears in the RHS of the DDE. That can be expanded using the real-valued Jacobi–Anger expansion:
$$\sin(R\cos\phi) = 2\sum_{n=0}^{\infty}(-1)^n J_{2n+1}(R)\cos((2n+1)\phi)$$

Where $J_n(x)$ is the nth Bessel function of the first kind.

Keeping only the fundamental ($n=0$):

$$
\begin{align*}
\sin(R\cos\phi) &\approx 2 J_1(R)\cos(\phi) \\
-k\sin(\theta(t-\tau)) &\approx -2k J_1(R)\cos(\omega t - \omega\tau)
\end{align*}
$$

Now, matching that to the LHS of the DDE, $d\theta/dt$, we get

$$
\begin{align*}
-R\omega \sin(\omega t) &= -2k J_1(R)\cos(\omega t - \omega\tau) \\
R\omega \sin(\omega t) &= 2k J_1(R)\cos(\omega t - \omega\tau) \\
\end{align*}
$$

Using $\cos(x-y) = \cos(x)\cos(y)+\sin(x)\sin(y)$,

$$
\begin{align*}
R\omega \sin(\omega t) &= 2k J_1(R) \big[\cos(\omega t)\cos(\omega\tau) + \sin(\omega t)\sin(\omega\tau) \big] \\
\end{align*}
$$

Matching coefficients of $\cos(\omega t)$:

$$ 
2kJ_1(R)\cos(\omega\tau) = 0 
$$

$$
\omega\tau = \pi/2
$$

This gives us $w=\pi/2\tau$: Omega is fixed by $\tau$ alone (independent of $k$) and furthermore, the time period $T=2\pi/\omega = 4\tau$ (confirming what we expected).

Next, matching the $\sin(\omega t)$ coefficients gives us:

$$
\begin{align*}
R\omega &= 2kJ_1(R)\sin(\omega\tau) \\
&= 2kJ_1(R)
\end{align*}
$$

With $\omega = \pi/(2\tau)$:

$$
\begin{align*}
R\,\frac{\pi}{2\tau} &= 2kJ_1(R)\sin\left(\frac{\pi\tau}{2\tau}\right) \\
\frac{J_1(R)}R &= \frac\pi{4k\tau}
\end{align*}
$$


$$\frac{2J_1(R)}{R} = \frac{\pi}{2k\tau}$$

Rearranging this equation, given the control parameter $k\tau$, we can calculate the amplitude $R$ to be

$$k\tau = \frac{\pi R}{4J_1(R)}$$

Saturation: $R \to j_{1,1} \approx 3.83$ rad as $k\tau \to \infty$.

#### Two-harmonics (useless copypasted stuff: don't take it seriously)

**Two-harmonic ansatz:** $\theta(t) = R_1\cos\omega t + R_3\cos 3\omega t$ and project the DDE onto each harmonic numerically. When $R_3=0$ you recover the $J_1$ equation exactly. 

The two-harmonic HB derivation in brief:

With $\theta(t) = R_1\cos\omega t + R_3\cos 3\omega t$ and $\omega\tau = \pi/2$, the delayed signal simplifies to:
$$\theta(t-\tau) = R_1\sin(\omega t) - R_3\sin(3\omega t)$$
(the phase-shift is $\pi/2$ for the fundamental, $3\pi/2$ for the 3rd harmonic, which flips its sign).

Projecting $\dot\theta = -k\sin(\theta(t-\tau))$ onto $\sin\omega t$ and $\sin 3\omega t$ gives:
$$R_1 = \frac{2k\tau}{\pi} \underbrace{\frac{1}{\pi}\int_0^{2\pi}\sin\!\big(R_1\sin\phi - R_3\sin 3\phi\big)\sin\phi\,d\phi}_{I_1}$$
$$R_3 = \frac{2k\tau}{3\pi} \underbrace{\frac{1}{\pi}\int_0^{2\pi}\sin\!\big(R_1\sin\phi - R_3\sin 3\phi\big)\sin 3\phi\,d\phi}_{I_3}$$

When $R_3=0$, $I_1 = 2J_1(R_1)$ exactly (from the Jacobi-Anger expansion), recovering the original equation.
