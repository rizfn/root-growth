## Delay differential equation for root growth

We use just a single parameter, $\theta$, to determine the angule of the root tip from the vertical. $\theta=0$ implies the root grows downwards. The entire behaviour of the root is assumed to be of the 'follow the leader' type, and thus the system can be described as the list of positions the tip traces out.

We use the following delay differential equation:

$$ \frac{\textrm{d}\theta}{\textrm{d}t} = \underbrace{- k \,\sin(\theta(t-\tau))}_{\text{Gravitropy}} + \underbrace{\eta\, \xi(t)}_{\text{Noise}} $$

The first time is the angular correction due to gravity sensing. If the root isn't vertical, say $\theta>0$, then the plant wants to correct it, and $\dot{\theta}$ is negative (hence the minus sign). However, as there's a delay between the growth and the sensing (due to the growth occuring at the division/elongation zones and not the tip itself) it uses the gravity sensed at a previous instant, delayed by time $\tau$.

### No noise, $\eta=0$

Depending on $\tau$ and $k$, the system shows either damped or sustained oscillations.

The delay differential equation is 

$$ \frac{\textrm{d}\theta}{\textrm{d}t} = - k \,\sin(\theta(t-\tau))$$

For small $\theta$, we can assume $\sin\theta\approx\theta$, converting it to a linear DDE:

$$ \frac{\textrm{d}\theta}{\textrm{d}t} = - k \,\theta(t-\tau)$$

Make the ansatz

$$ \theta(t) = A\, e^{\lambda t} $$

Then, 

$$ 
\begin{align*}
\frac{\textrm{d}\theta}{\textrm{d}t} &= A\lambda\, e^{\lambda t} \\
\theta(t-\tau) &= A\,e^{\lambda(t-\tau)} \\
&= A\,e^{\lambda t} \,e^{-\lambda \tau}
\end{align*}
$$

Substituting in,

$$ 
\begin{align*}
\frac{\textrm{d}\theta}{\textrm{d}t} &= - k \,\theta(t-\tau) \\
A\,\lambda\, e^{\lambda t} &= -k \, A\,e^{\lambda t} \,e^{-\lambda \tau} \\
\lambda &= -k\,e^{-\lambda \tau}
\end{align*}
$$

Which gives us the characteristic equation

$$\lambda + k\,e^{-\lambda \tau} = 0$$

This is a transcendental equation, that needs to be solved for $\lambda$. $\lambda$ is, in principle, complex, and thus it can be written as 

$$ \lambda = a + bi$$

where $a$ controls the amplitude of the oscillations. If $a>0$, the oscillations amplify, while if $a<0$, they're damped. Thus, the critical boundary can be found to be where $a=0$:

$$ 
\begin{align*}
b\,i + k\,e^{-b\,i\, \tau} &= 0 \\
b\,i + k\,(\cos(b\,\tau) - i \sin(b\,\tau)) &= 0
\end{align*}
$$

The real part gives us:

$$ k \, \cos(b\,\tau)=0 $$
Which implies that
$$b\tau = \frac\pi2 + n\pi ;\qquad n=\{0,1,2,3,..\}$$

The imaginary part gives us

$$
b - k\, \sin(b\,\tau)=0 \\
b = k
$$

Thus, the critical boundary occurs at 
$$ k \cdot\tau = \frac\pi2 $$

Note that $b$ represents the angular velocity ($\omega$). The time period can thus be written as:

$$
\begin{align*}
T &= \frac{2\,\pi}{b} \\
&= \frac{2\,\pi}{k} \\
&= 2\,\pi \left(\frac{2\,\tau}{\pi}\right) \\
&= 4\, \tau
\end{align*}
$$

If we (handwavingly) move from time to space, the time period transforms to the wavelength. It's clear that $\tau$ comes about due to a delay across the growth zone and the tip, and thus it should transform to the distance between the root tip and the elongation zone.


### Non-zero noise, $\eta \neq 0$

Again, let's use the linear stochastic delay differential equation:

$$ d\theta = - k \,\theta(t-\tau)\; dt + \eta \; dW(t)$$

where $W(t)$ is a weiner process with mean 0 and variance $dt$.

Ito's lemma, applied to $f(x)$, states that:

$$ df = \frac{\partial f}{\partial x} dx + \frac12 \frac{\partial^2f}{\partial x^2} (dx)^2$$

Applying this to $f(\theta) = \theta^2$,

$$ d(\theta^2) = 2\theta \, d\theta + \frac{1}{2}(2)(d\theta)^2 $$

Since we have $d\theta = - k \,\theta(t-\tau)\; dt + \eta \; dW$, we can calculate

$$ (d\theta)^2 = \left[- k \,\theta(t-\tau)\; dt + \eta \; dW\right]^2 $$

Note that we get the following terms (and as $dW\sim\mathcal{O}(\sqrt{dt})$)
- $(dt)^2\sim\mathcal{O}(dt^2)$
- $dt\cdot dW \sim\mathcal{O}(dt^{3/2})$
- $(dW)^2 \sim\mathcal{O}(dt)$

Thus, we can neglect the higher order terms, and only keep terms up to $\mathcal{O}(dt)$:

$$ 
\begin{align*}
(d\theta)^2 &=  \eta^2 \; dW^2 \\
&= \eta^2 \;dt
\end{align*} $$

Putting that in:

$$ 
\begin{align*}
d(\theta^2) &= 2\theta[-k\theta(t-\tau)dt + \eta dW] + \eta^2 dt \\
&= -2k\theta(t) \theta(t-\tau) dt + 2 \theta(t) \eta dW + \eta^2dt 
\end{align*} 
$$

Taking expectations,

$$ \frac{d\langle\theta^2\rangle}{dt} = -2k\langle\theta(t)\theta(t-\tau)\rangle + \eta^2 $$

We have the autocorrelation at a time delta of $\tau$ in here. But it's also interesting to study the general autocorrelation

$$ C(\Delta) = \langle\theta(t)\theta(t-\Delta)\rangle $$

Intuitively, in a noisy system, you should be correlated locally, but over very long times the noise can affect the oscillations, perturbing them. Thus, the autocorrelation function should decrease.

Substitute $u=t-\Delta$, so we get

$$ C(\Delta) = \langle\theta(t)\theta(u)\rangle $$

At the 'steady state' (where $t$ itself doesn't matter, but only $\Delta$), we can take the derivative with respect to $\Delta$, and knowing that $du = - d\Delta$,

$$
\begin{align*}
\frac{\partial C(\Delta)}{\partial(\Delta)} &= \frac{\partial}{\partial(u)}\langle\theta(t)\theta(u)\rangle \cdot \frac{du}{d\Delta} \\
&= - \frac{\partial}{\partial(u)}\langle\theta(t)\theta(u)\rangle
\end{align*}
$$

As the past value is 'fixed', we end up with

$$
\begin{align*}
\frac{\partial C(\Delta)}{\partial(\Delta)} &= - \left\langle\theta(t) \frac{\partial \theta(u)}{\partial(u)}\right\rangle
\end{align*}
$$

Now, we can use our SDDE, as we know

$$ \frac{d\theta(u)}{du} = -k\theta(u-\tau) + \eta\xi(u) $$

Then,

$$
\begin{align*}
\frac{dC(\Delta)}{d\Delta} &= -\left\langle \theta(t) \big[-k\theta(u-\tau) + \eta\xi(u)\big] \right\rangle \\
&= k\langle \theta(t) \theta(u-\tau) \rangle - \eta\langle \theta(t) \xi(u) \rangle
\end{align*}
$$

Move back to using $\Delta$, with $u=t-\Delta$

$$
\begin{align*}
\frac{dC(\Delta)}{d\Delta} &= k\langle \theta(t) \theta(t-\Delta-\tau) \rangle - \eta\langle \theta(t) \xi(t-\Delta) \rangle \\
&= k\, C(\Delta + \tau) - \eta\big\langle \theta(t) \xi(t-\Delta) \big\rangle
\end{align*}
$$

There are two terms: one, that looks similar to the ODE that we had earlier (the autocorrelation has it's own damping or growing based on $\tau$ and $k$), and the other which is a noise-induced damping.
