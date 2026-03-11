## Other response functions

The period 4 is believed to be somewhat 'universal' to delayed control systems.

If we have a delayed response of the form

$$ \frac{\mathrm{d}\theta}{\mathrm{d}t} = -k f(\theta(t-\tau)) $$

$f(x) = x$ blows up after the Hopf bifurcation, so we need functions that grow slower. We consider three examples:

- $f(x) = \ln(x+1)$, grows slower than linear
- $f(x) = \tanh(x)$, grows, slows down, and saturates at 1 (asymptotic to $y=1$)
- $f(x) = x e^{-x}$, grows, peaks, decreases, and is asymptotic to the x-axis ($y=0$)

Note that we require the functions to be **odd** functions. $\tanh(x)$ is already odd, but we can convert the other two to be odd by using the sign function $\mathrm{sgn}(x)$ and the absolute value $|x|$. Thus, we use:

$$f(x)=\mathrm{sgn}(x) \ln(\lvert x \rvert + 1)$$

$$f(x)=\tanh(x)$$

$$f(x)=\mathrm{sgn}(x)\lvert x\rvert e^{-\lvert x\rvert}$$