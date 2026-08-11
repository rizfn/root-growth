# Where the damping of $C(\Delta)$ comes from

This note explains, from scratch, why the autocorrelation of the noisy delayed
oscillator decays exponentially, why the decay rate is proportional to $\eta^2$,
and what the quantity $\tfrac12\langle Z^2\rangle$ plotted in the third panel of
the damping figure actually is.

Everything below uses only: a Taylor expansion, an average of $\cos^2$ over a
cycle, and one standard series (which is where the Bessel function $J_1$ comes
from). No dynamical-systems vocabulary is assumed.

The model is

$$\frac{d\theta}{dt} \;=\; -k\,\sin\bigl(\theta(t-\tau)\bigr) \;+\; \eta\,\xi(t),$$

where $\xi$ is white noise: $\langle\xi(t)\rangle=0$ and
$\langle\xi(t)\xi(t')\rangle=\delta(t-t')$. Time is measured in units of the
delay, so we may set $\tau=1$ wherever it is convenient and restore it at the
end.

---

## 1. The noiseless oscillation, and where $k\tau=\dfrac{\pi R}{4J_1(R)}$ comes from

Set $\eta=0$ for now. We look for an oscillation that is a pure sine wave,

$$\theta_0(t) \;=\; R\,\sin(\omega t),$$

and we will find both the frequency $\omega$ and the amplitude $R$ by demanding
that this actually solves the equation.

**The frequency.** Put the guess into the delayed term:

$$\theta_0(t-\tau) \;=\; R\sin(\omega t-\omega\tau).$$

Now here is the special feature of this system. Suppose the delay is exactly a
quarter of a period, i.e. $\omega\tau=\pi/2$. Then

$$\theta_0(t-\tau) \;=\; R\sin\!\left(\omega t-\tfrac{\pi}{2}\right) \;=\; -R\cos(\omega t),$$

so the *delayed* signal is the *derivative-like* partner of the original one: a
sine has become a cosine. That is exactly what an oscillator needs — a restoring
term that is a quarter-cycle out of step with the displacement. So we expect

$$\omega=\frac{\pi}{2\tau},\qquad\text{period } T=\frac{2\pi}{\omega}=4\tau .$$

This is the origin of the period-$4\tau$ cycle: the delay supplies the quarter-cycle
lag that an oscillation requires.

**The amplitude.** Substitute back. The left-hand side is

$$\dot\theta_0 \;=\; R\,\omega\cos(\omega t),$$

and the right-hand side is

$$-k\sin\bigl(\theta_0(t-\tau)\bigr) \;=\; -k\sin\bigl(-R\cos\omega t\bigr) \;=\; k\,\sin\bigl(R\cos\omega t\bigr).$$

We now need to know what $\sin(R\cos x)$ looks like as a sum of cosines. This is
a standard series,

$$\sin(z\cos x)\;=\;2\Bigl[J_1(z)\cos x \;-\; J_3(z)\cos 3x \;+\; J_5(z)\cos 5x-\dots\Bigr].$$

You do not need to know anything about Bessel functions to use this. $J_1(z)$ is
simply *the amount of plain $\cos x$ contained in the wiggly function
$\sin(z\cos x)$*, and for small $z$ it behaves as $J_1(z)\approx z/2$, which just
reproduces $\sin(z\cos x)\approx z\cos x$.

So the right-hand side is

$$2k\,J_1(R)\cos(\omega t) \;-\; 2k\,J_3(R)\cos(3\omega t)+\dots$$

Matching the $\cos(\omega t)$ term on both sides — the term that oscillates at the
frequency we care about — gives

$$R\,\omega \;=\; 2k\,J_1(R).$$

Putting in $\omega=\pi/2\tau$ and rearranging:

$$\boxed{\;k\tau \;=\; \frac{\pi R}{4\,J_1(R)}\;}$$

This is the amplitude relation. Given the single combination $k\tau$, it fixes
the size $R$ of the oscillation. Two sanity checks:

* For a small orbit, $J_1(R)\to R/2$, so $k\tau\to \pi/2$. This is the threshold:
  below $k\tau=\pi/2$ there is no oscillation at all, and just above it the
  oscillation is born with vanishing amplitude.
* As $k\tau$ increases, $R$ grows. At $k\tau=2$ it gives $R=1.36$; at $k\tau=3$,
  $R=2.15$.

The terms we threw away ($J_3$, $J_5$, …) are the higher harmonics: the real
orbit is not *exactly* a sine wave. They are small when $R$ is small, and they
are the reason the closed-form answer in §7 slowly degrades as $k\tau$ grows.

---

## 2. What the noise can do: two very different things

Turn the noise back on. At any moment, a small random kick changes $\theta$ by a
small amount. Split its effect into two parts:

1. **It can change the size of the loop.** But the oscillation is an *attractor*:
   push the amplitude away from $R$ and it comes back. So this effect does not
   accumulate — it stays bounded forever.

2. **It can change the timing of the loop.** Nothing pushes the timing back.
   The equation has no clock in it; a solution shifted in time,
   $\theta_0(t+\text{const})$, is just as good a solution as $\theta_0(t)$. So a
   timing error, once made, is permanent, and the next kick adds to it.

This asymmetry is the whole story. Amplitude errors stay small; **timing errors
accumulate**, performing a random walk. That accumulating timing error is what
destroys the correlation between $\theta$ now and $\theta$ a long time later.

We describe the timing by a phase. Write the noisy solution as

$$\theta(t)\;\approx\;R\,\sin\bigl(\omega t+\psi(t)\bigr),$$

where $\psi$ is a slowly wandering phase. If $\psi$ were constant we would have a
perfect clock. Instead $\psi$ random-walks.

---

## 3. The phase response $Z$: how much does one kick shift the clock?

Ask a concrete question. The system is running along its cycle. At the moment
when the phase is $\phi$, give $\theta$ a single small nudge of size $\delta$.
Wait a long time, until the amplitude has relaxed back to $R$. Compare the
oscillation with the one you would have had without the nudge. They now have the
same shape and the same frequency, but they are **offset in time**.

Convert that permanent time offset into a phase offset, $\Delta\psi$. It is
proportional to the size of the nudge (because the nudge is small), so

$$\Delta\psi \;=\; Z(\phi)\,\delta .$$

The proportionality factor $Z(\phi)$ is the **phase response**. It depends on
*when* in the cycle you kicked — that is the whole point. It is a pure number: a
phase shift per unit of $\theta$.

$Z$ is not something we have to guess; it is measured directly by doing exactly
what the definition says (`prc.cpp` kicks the converged orbit at 200 phases and
records the resulting time shift). Two features matter:

* **Its shape.** For an orbit that is close to a sine wave, $Z(\phi)$ is itself
  close to a single sinusoid, $Z(\phi)\approx Z_0\cos(\phi+\chi)$: there is one
  moment in the cycle where a kick shifts the clock most, and half a period later
  it shifts it equally in the other direction.
* **Its size.** $Z_0$ scales as $1/R$. This is the important physical point and
  it is easy to see: to move a *large* loop sideways in time you must move it a
  long way in $\theta$, so a fixed nudge $\delta$ buys you a small phase shift.
  A tiny loop is easy to shove around, so $Z_0$ is large.

Everything about the noise now enters through this one function.

---

## 4. The phase performs a random walk

The noise is a continuous stream of independent small kicks. Between $t$ and
$t+dt$ the kick is $\eta\,\xi(t)\,dt$, so the phase obeys

$$\frac{d\psi}{dt}\;=\;\eta\,Z(\phi)\,\xi(t).$$

Because successive kicks are independent, the *variance* of $\psi$ adds up. Over
a time $t$,

$$\operatorname{Var}\bigl[\psi(t)-\psi(0)\bigr]
\;=\;\eta^2\!\int_0^t \! Z\bigl(\phi(t')\bigr)^2\,dt' .$$

The system runs many times around the cycle, so $Z^2$ can be replaced by its
average over one cycle, $\langle Z^2\rangle$. Hence

$$\operatorname{Var}\bigl[\psi(t)-\psi(0)\bigr]\;=\;\eta^2\,\langle Z^2\rangle\,t .$$

This is a random walk: variance growing linearly in time. Writing it in the
conventional form $\operatorname{Var}=2Dt$ identifies the **phase diffusion
coefficient**

$$\boxed{\;D\;=\;\tfrac12\,\eta^2\,\langle Z^{2}\rangle\;}$$

There is the $\tfrac12\langle Z^2\rangle$. It is nothing more than: *the average
squared sensitivity of the clock to a kick, halved because variance $=2Dt$.*

Note immediately that $D\propto\eta^2$, with no threshold and no special noise
level. Doubling the noise quadruples the rate at which the clock loses track.

---

## 5. From a wandering phase to a decaying autocorrelation

Now compute what we actually measure. With $\theta(t)=R\sin(\omega t+\psi(t))$,

$$C(\Delta)\;=\;\bigl\langle\theta(t)\,\theta(t+\Delta)\bigr\rangle
\;=\;R^2\bigl\langle \sin(\omega t+\psi)\,\sin(\omega t+\omega\Delta+\psi')\bigr\rangle,$$

with $\psi=\psi(t)$ and $\psi'=\psi(t+\Delta)$. In practice $C(\Delta)$ is measured
by sliding a window along a long record: $\langle\cdot\rangle$ is an average over
the window position $t$, at fixed lag $\Delta$.

Use $\sin A\sin B=\tfrac12[\cos(A-B)-\cos(A+B)]$:

$$C(\Delta)=\frac{R^{2}}{2}\Bigl[\bigl\langle\cos(\omega\Delta+\psi'-\psi)\bigr\rangle
-\bigl\langle\cos(2\omega t+\omega\Delta+\psi+\psi')\bigr\rangle\Bigr].$$

Now look at where $t$ appears in each one.

**The first term survives.** The only $t$ in it is inside
$\psi'-\psi=\psi(t+\Delta)-\psi(t)$, the phase drift during the lag. Sliding the
window does not change how big that drift typically is, so every window position
contributes the same thing.

**The second term averages away.** Here $t$ appears on its own, as $2\omega t$,
which turns twice as fast as the oscillation. Slide the window forward by half a
period, $\pi/\omega$: then $2\omega t$ grows by $2\pi$ — one whole turn of the
cosine — while $\psi$ and $\psi'$ have barely moved. So across that stretch the
cosine passes through every angle once, and the positive and negative parts
cancel.

Written out, with the slow parts held at some fixed value $c$ and averaging over
one period $T=2\pi/\omega$:

$$\frac{1}{T}\int_{0}^{T}\!\cos(2\omega t+c)\,dt
=\frac{1}{T}\left[\frac{\sin(2\omega t+c)}{2\omega}\right]_{0}^{T}
=\frac{\sin(c+4\pi)-\sin(c)}{2\omega T}=0,$$

because $2\omega T=4\pi$, and $\sin$ repeats every $2\pi$. Every period of the
record contributes zero in this way, so the whole record does.

Holding $\psi$ fixed across one period is safe: it drifts by only a few percent of
a turn per period at the noise levels used here. And that drift can only help —
it spreads the angle around further, it cannot undo the cancellation.

What is left is

$$C(\Delta)=\frac{R^{2}}{2}\bigl\langle\cos\bigl(\omega\Delta+\psi'-\psi\bigr)\bigr\rangle .$$

The drift $\psi'-\psi$ is a sum of very many independent small kicks, so it is
Gaussian with mean zero, and by §4 its variance is $2D\Delta$. For any Gaussian
$u$ with mean zero and variance $\sigma^{2}$,
$\langle\cos(a+u)\rangle=\cos(a)\,e^{-\sigma^{2}/2}$. Therefore

$$\boxed{\;C(\Delta)\;=\;\frac{R^{2}}{2}\,\cos(\omega\Delta)\;e^{-D\Delta},
\qquad D\;=\;\tfrac12\,\eta^{2}\langle Z^{2}\rangle\;}$$

This is the entire content of the first two panels of the figure:

* the autocorrelation **oscillates** at the frequency of the cycle, $\omega=\pi/2\tau$
  (period $4\tau$),
* its envelope **decays exponentially**, and
* the decay rate is **exactly** the phase diffusion coefficient.

So the third panel plots $c \equiv D/\eta^{2}=\tfrac12\langle Z^{2}\rangle$:
the damping rate stripped of its trivial $\eta^2$ dependence, leaving a number
that depends only on the shape of the orbit, i.e. only on $k\tau$.

---

## 6. Why the decay is *exponential*, and why there is no threshold

Two things are worth pausing on, because the original manuscript got both wrong.

**Exponential, not something else.** The decay is $e^{-D\Delta}$ purely
because the variance of the accumulated phase grows *linearly* in $\Delta$, and
because a Gaussian phase enters $\cos$ through $e^{-\sigma^2/2}$. Linear growth
of variance is the signature of independent increments — it requires nothing more
than that the noise is white.

**No threshold.** $D=\tfrac12\langle Z^2\rangle\eta^2$ holds for any $\eta$
small enough that the orbit is not destroyed. There is no noise level at which
something switches on. A plateau at small $\eta$ followed by a sudden rise is not
something this mechanism can produce, and the plateau in the original figure was
an artefact of the fitting procedure, not physics. (A genuinely sharp threshold
does exist in this system, but it comes from a completely different mechanism —
noise-driven escape between two coexisting cycles at $k\tau\approx4$ — and it is
exponential in $1/\eta^2$, not a power law.)

---

## 7. A closed form for small orbits, and its limits

Combining §3 ($Z_0\propto 1/R$) with §1 ($R$ from the amplitude relation) gives a
prediction for $c(k\tau)$ with no free parameters. For a nearly sinusoidal orbit,

$$Z(\phi)\;\approx\;\frac{4}{\pi R}\cos(\phi+\chi)
\qquad\Longrightarrow\qquad
\langle Z^{2}\rangle=\frac{Z_0^{2}}{2}=\frac{8}{\pi^{2}R^{2}},$$

so that

$$\boxed{\;c\;=\;\frac{D}{\eta^{2}}\;\approx\;\frac{4}{\pi^{2}R^{2}},
\qquad k\tau=\frac{\pi R}{4J_1(R)}\;}$$

> **Honesty about one step.** The $1/R$ *scaling* of $Z_0$ follows from the
> argument in §3. The numerical prefactor $4/\pi$ does not follow from anything
> as simple. For an ordinary oscillator one fixes it by a normalisation condition
> on the orbit; for a *delay* equation the state of the system is not a point but
> the entire history $\theta(s)$ for $s\in[t-\tau,t]$, and the corresponding
> condition acquires an extra memory integral over that history. My attempts to
> evaluate that integral by elementary means gave answers differing from the
> measured value by factors of $\omega\tau=\pi/2$, so the constant $4/\pi$ quoted
> here is the one that **matches the numerically computed $Z$**, not one I have
> derived from scratch. Treat §1–§6 as derived and this prefactor as fitted.

How well does it do? Comparing with $\tfrac12\langle Z^2\rangle$ computed from the
measured $Z$:

| $k\tau$ | $R$ | $4/\pi^2R^2$ | measured $\tfrac12\langle Z^2\rangle$ | error |
|---|---|---|---|---|
| 1.6 | 0.382 | 2.76 | 2.76 | 0.1 % |
| 2.0 | 1.362 | 0.219 | 0.232 | 6 % |
| 3.0 | 2.149 | 0.088 | 0.135 | 54 % |

The formula is essentially exact just above threshold and deteriorates steadily.
The reason is the harmonics we discarded in §1: as $R$ grows, the orbit stops
being a sine wave, $Z$ stops being a single cosine, and $\langle Z^2\rangle$ picks
up contributions the formula knows nothing about. This is precisely why the
figure plots $\tfrac12\langle Z^2\rangle$ from the numerically computed $Z$ rather
than the closed form.

---

## 8. Reading the shape of the $c(k\tau)$ curve

With the above, the curve in the third panel becomes readable:

* **It diverges as $k\tau\to\pi/2$.** At threshold the orbit is born with zero
  amplitude, $R\to0$, and $Z_0\sim1/R\to\infty$. A vanishingly small loop can be
  shoved arbitrarily far in phase by a fixed kick, so its clock is hopeless. This
  is a real physical statement: oscillations just above onset are very poor
  clocks.

  The divergence has a definite power. Expand the amplitude relation for small
  $R$ using $J_1(R)\approx R/2-R^3/16$:

  $$k\tau=\frac{\pi R}{4J_1(R)}=\frac{\pi}{2-R^{2}/4}
  \approx\frac{\pi}{2}\left(1+\frac{R^{2}}{8}\right)
  \quad\Longrightarrow\quad
  R^{2}=\frac{16}{\pi}\left(k\tau-\frac{\pi}{2}\right).$$

  Since $c\approx 4/\pi^{2}R^{2}$, this gives

  $$c\;\approx\;\frac{1}{4\pi\,\bigl(k\tau-\pi/2\bigr)},$$

  so the prefactor diverges as a simple inverse power of the distance from the
  bifurcation. Against the numerically computed $Z$ this is accurate to $1.4\%$ at
  $k\tau=1.60$ and $5\%$ at $k\tau=1.65$, degrading beyond as the expansion of
  $R^{2}$ loses accuracy (keeping the exact $R$ instead does much better: the
  $4/\pi^{2}$ form stays within $2\%$ out to $k\tau=1.8$).

* **It falls like $1/R^{2}$ over $1.6\lesssim k\tau\lesssim2.8$.** The orbit grows,
  and a bigger loop is a better clock.

* **It turns around near $k\tau\approx2.8$ and rises again.** Here $1/R^2$ is no
  longer the whole story. The orbit has become distinctly non-sinusoidal, with
  slow stretches where it lingers; a kick delivered during a slow stretch buys a
  disproportionately large timing error, so $\langle Z^2\rangle$ grows again even
  though $R$ is still increasing.

* **It ends at $k\tau\approx3.85$**, where the period-$4\tau$ cycle stops existing.

---

## Summary in one line

Noise cannot change the size of the loop for long, but it permanently changes its
timing; timing errors accumulate as a random walk with diffusion constant
$D=\tfrac12\eta^2\langle Z^2\rangle$; a diffusing phase turns the autocorrelation
into $\tfrac{R^2}{2}\cos(\omega\Delta)e^{-D\Delta}$; hence the damping rate is
$D=\tfrac12\langle Z^2\rangle\,\eta^2$, proportional to $\eta^2$ always, with
a prefactor set entirely by the size and shape of the noiseless orbit — which the
amplitude relation $k\tau=\pi R/4J_1(R)$ delivers from $k\tau$ alone.
