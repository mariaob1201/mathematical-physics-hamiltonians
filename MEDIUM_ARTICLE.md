# When Physics Meets Statistics: Hamiltonian Monte Carlo from Scratch

*How classical mechanics gives us one of the most powerful tools in modern Bayesian data science*

---

There is a beautiful coincidence hiding in plain sight between two fields that rarely share a sentence: **classical mechanics** and **Bayesian statistics**.

In mechanics, a Hamiltonian describes how a physical system evolves through time — how a planet orbits a star, or how a pendulum swings. In statistics, we need to explore a probability distribution over model parameters — something that grows exponentially harder as the number of parameters increases.

The insight that connects them, formalized by Radford Neal in 2011, is this: **you can pretend your probability distribution is a physical landscape and roll a ball across it**. The ball's trajectory will naturally explore the landscape with remarkable efficiency, visiting high-probability regions more often than low-probability ones — exactly what a sampler needs to do.

This is Hamiltonian Monte Carlo (HMC). It is the algorithm that powers Stan, PyMC, and NumPyro. It is why modern Bayesian computation is tractable. And in this article, we will build it from scratch.

---

## The Problem: Exploring a Posterior

Suppose you have binary medical data — say, which tumours are malignant versus benign — and you want to fit a logistic regression model. In a Bayesian framework, you do not get a single answer. You get a **posterior distribution** over every model coefficient:

$$
\pi(\beta \mid X, y) \propto \underbrace{\prod_{i=1}^n \sigma(x_i^\top \beta)^{y_i}[1-\sigma(x_i^\top \beta)]^{1-y_i}}_{\text{likelihood}} \cdot \underbrace{\mathcal{N}(\beta; 0, \sigma^2 I)}_{\text{prior}}
$$

This distribution lives in a high-dimensional space. With 5 features and an intercept, that is 6 dimensions. With 30 features, 31. The posterior has complex correlations, curved ridges, and no closed form.

You cannot integrate it analytically. You need to **sample** from it.

The naive approach — a random-walk Metropolis-Hastings sampler — takes tiny, undirected steps through parameter space. It is like exploring a mountain range by walking in a random direction for a few metres at a time. You will get there, but it will take a long time and the samples will be heavily autocorrelated.

HMC takes a different approach: **it gives you a physics simulation**.

---

## The Hamiltonian Framework

Define a total energy function over positions $q$ (our parameters $\beta$) and momenta $p$ (an auxiliary variable we will introduce):

$$
H(q, p) = \underbrace{U(q)}_{\text{potential}} + \underbrace{K(p)}_{\text{kinetic}}
$$

where

$$
U(q) = -\log \pi(q \mid X, y) \qquad \text{(negative log-posterior)}
$$

$$
K(p) = \frac{1}{2} p^\top M^{-1} p \qquad \text{(Gaussian kinetic energy)}
$$

Hamilton's equations of motion then give us:

$$
\frac{dq}{dt} = \frac{\partial H}{\partial p} = M^{-1} p
\qquad
\frac{dp}{dt} = -\frac{\partial H}{\partial q} = -\nabla U(q)
$$

The key property: **Hamiltonian dynamics conserves $H$**. A particle obeying these equations stays on a surface of constant energy. This means it moves through parameter space without climbing too high or falling into a valley — it glides along the contours of the posterior.

---

## Why This Gives a Valid MCMC Sampler

Three properties of Hamiltonian dynamics make this work as a Monte Carlo method:

**1. Reversibility.** Hamilton's equations are time-reversible. If we negate the momentum, the trajectory runs backwards. This is required for detailed balance, which guarantees the correct stationary distribution.

**2. Volume preservation (Liouville's theorem).** The flow defined by Hamilton's equations is symplectic: it preserves the volume of any region in phase space. This means the Jacobian of the transformation is 1, so we do not need to compute a correction factor in the Metropolis step.

**3. Energy conservation.** If we could simulate the dynamics exactly, the Metropolis acceptance probability would always be 1. We cannot — we use a numerical integrator — but energy conservation means acceptance rates stay high.

---

## The Leapfrog Integrator

We simulate Hamilton's equations numerically using the **leapfrog (Störmer-Verlet) integrator**:

```
p ← p − (ε/2) ∇U(q)          # half step for momentum
for k = 1, …, L−1:
    q ← q + ε M⁻¹ p           # full step for position
    p ← p − ε ∇U(q)           # full step for momentum
q ← q + ε M⁻¹ p               # final position step
p ← p − (ε/2) ∇U(q)          # final half step for momentum
```

This is not an arbitrary choice. The leapfrog scheme is **symplectic**: it exactly preserves a modified Hamiltonian $\tilde{H}(q,p) = H(q,p) + O(\varepsilon^2)$. This means:

- Phase-space volume is preserved exactly (not approximately)
- The Metropolis correction remains valid even with finite step sizes
- Energy error is bounded, not accumulated — it stays near $O(\varepsilon^2)$, never drifts

Compare this to a standard Euler integrator, which would accumulate energy error at every step and eventually produce samples from the wrong distribution.

```python
def leapfrog(q, p, grad_U, epsilon, L, M_inv=None):
    if M_inv is None:
        M_inv = np.eye(len(q))
    q, p = q.copy(), p.copy()

    p -= 0.5 * epsilon * grad_U(q)
    for _ in range(L - 1):
        q += epsilon * (M_inv @ p)
        p -= epsilon * grad_U(q)
    q += epsilon * (M_inv @ p)
    p -= 0.5 * epsilon * grad_U(q)

    return q, p
```

---

## The Full HMC Transition

One complete HMC step:

1. **Refresh momentum.** Sample $p \sim \mathcal{N}(0, M)$ independently. This breaks the determinism and ensures ergodicity.

2. **Simulate dynamics.** Run $L$ leapfrog steps from $(q, p)$ to get a proposal $(q^*, p^*)$.

3. **Accept or reject.** Apply the Metropolis criterion:

$$
\alpha = \min\!\left(1,\; \exp\bigl(H(q,p) - H(q^*, p^*)\bigr)\right)
$$

If the leapfrog integration were exact, $H$ would be perfectly conserved and $\alpha = 1$ always. In practice, $\alpha$ is very close to 1 for a well-tuned step size — typically 60–90%.

```python
def hmc_step(q, X, y, epsilon, L, sigma_prior=1.0):
    d = len(q)
    p = np.random.randn(d)                          # refresh momentum

    H_current  = potential_energy(q, X, y, sigma_prior) + 0.5 * np.dot(p, p)
    q_prop, p_prop = leapfrog(q, p, lambda b: grad_potential(b, X, y, sigma_prior),
                               epsilon, L)
    H_proposed = potential_energy(q_prop, X, y, sigma_prior) + 0.5 * np.dot(p_prop, p_prop)

    if np.log(np.random.uniform()) < H_current - H_proposed:
        return q_prop, True
    return q, False
```

---

## The Statistical Hamiltonian for Logistic Regression

For Bayesian logistic regression with a Gaussian prior $\beta \sim \mathcal{N}(0, \sigma^2 I)$:

$$
U(\beta) = -\sum_{i=1}^n \bigl[y_i \log \sigma(x_i^\top \beta) + (1-y_i)\log(1-\sigma(x_i^\top \beta))\bigr] + \frac{\|\beta\|^2}{2\sigma^2}
$$

The analytic gradient is:

$$
\nabla U(\beta) = X^\top (\sigma(X\beta) - y) + \frac{\beta}{\sigma^2}
$$

This is the negative gradient of the log-posterior — identical in form to the gradient used in logistic regression training, plus a regularisation term from the prior. No automatic differentiation needed; the formula is clean and fast.

---

## Applied to Breast Cancer Data

We apply HMC to the **Breast Cancer Wisconsin** dataset (569 samples, binary outcome: malignant vs benign). We select the 5 most informative features by mutual information and add an intercept, giving a 6-dimensional posterior.

**Hyperparameters:**
- Step size $\varepsilon = 0.08$
- Leapfrog steps $L = 30$
- 600 warmup draws (discarded)
- 2 000 posterior draws

```
  Warmup  (600 steps, eps=0.08, L=30) ...
  Warmup acceptance rate : 87.3%
  Sampling (2000 draws) ...
  Sampling acceptance rate: 85.1%
```

An 85% acceptance rate is right in the target range. High acceptance with large $L$ means the sampler is taking long, efficient leaps through the posterior.

### What the posterior tells us

Rather than a single coefficient estimate, each parameter now has a full marginal distribution with credible intervals. The pair plots reveal that some features are negatively correlated in the posterior — the model knows that if one coefficient is large, another should be smaller to explain the same data.

This is information that maximum likelihood estimation cannot give you.

### Calibration and uncertainty

The posterior predictive probability for a new observation is:

$$
p(y^* = 1 \mid x^*, X, y) \approx \frac{1}{S} \sum_{s=1}^S \sigma\bigl(x^{*\top} \beta^{(s)}\bigr)
$$

Averaging over posterior draws produces probabilities that are **well-calibrated by construction**: the spread of the posterior translates directly into predictive uncertainty. Observations near the decision boundary get wide credible intervals; clearly benign or malignant cases get tight, confident predictions.

---

## Diagnostics: How to Know It Worked

HMC is only useful if the chain has converged. Three diagnostics matter:

**Trace plots** should look like stationary noise — no trends, no slow drifts. The chain should explore the same region from iteration 1 to iteration 2000.

**Effective Sample Size (ESS)** corrects for autocorrelation. If 2000 draws have an ESS of 1800, each draw is nearly independent. If ESS is 50, your chain is moving slowly and you need more samples or a better step size.

$$
\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}
$$

**Energy conservation** plots $H$ along the last leapfrog trajectory. Large oscillations in $H$ mean $\varepsilon$ is too large — reduce it. A flat $H$ means energy is well-conserved and the Metropolis correction is near-trivial.

---

## HMC vs Random Walk: Why It Matters

A random-walk Metropolis sampler in 6 dimensions with a well-tuned step size will typically achieve an ESS of 5–15% of total draws. HMC routinely achieves 70–90%.

The reason is geometric. A random walk in $d$ dimensions takes $O(d^2)$ steps to travel a distance $O(d)$. HMC takes $O(d^{5/4})$ steps (Neal 2011) because it uses gradient information to move in directions that are actually productive.

As the number of parameters grows, this advantage compounds. At 100 parameters, HMC is not just better — random-walk sampling becomes practically infeasible.

---

## What Liouville's Theorem Is Really Saying

Liouville's theorem states that for any Hamiltonian system, the volume of a region in phase space is conserved under the flow. In sampling terms: if we start with a set of initial conditions distributed according to $\exp(-H)$, after simulating Hamiltonian dynamics for any duration, those conditions remain distributed according to $\exp(-H)$.

This means Hamiltonian dynamics is not just a heuristic for exploring distributions — it is a **structure-preserving map** on probability space. The leapfrog integrator inherits this property because it is symplectic: it exactly preserves the discrete-time analogue of Liouville's theorem.

When you read that HMC "has the correct stationary distribution," this is why. It is not an approximation that gets better with more samples. It is exact.

---

## Conclusion

Hamiltonian Monte Carlo is not an analogy between physics and statistics. It is an application of physics to statistics. The conservation laws, the symplectic geometry, the leapfrog integrator — all of it is doing real mathematical work, and the guarantees are exact.

What makes HMC remarkable is that it turns a fundamental theorem of classical mechanics (conservation of phase-space volume) into a free lunch for sampling: large, directed proposals that are accepted at high rates, without any correction for the transformation Jacobian.

The implementation is not complex. The math is not approximate. And the result — a fully calibrated posterior distribution over every model parameter — is something that maximum likelihood estimation cannot produce, no matter how large the dataset.

The code for this article, including a full Jupyter notebook, is available at:
[github.com/mariaoros/mathematical-physics-hamiltonians](https://github.com/mariaoros/mathematical-physics-hamiltonians)

---

## References

- Neal, R. M. (2011). MCMC Using Hamiltonian Dynamics. *Handbook of Markov Chain Monte Carlo*, Chapman & Hall.
- Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434*.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics*. Cambridge University Press.
- Geyer, C. J. (1992). Practical Markov Chain Monte Carlo. *Statistical Science*, 7(4), 473–483.

---

*All code in this article is written in Python using NumPy only — no probabilistic programming frameworks. The full implementation including diagnostics, phase-space visualisations, and calibration plots is in the linked repository.*
