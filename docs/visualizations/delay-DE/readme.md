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