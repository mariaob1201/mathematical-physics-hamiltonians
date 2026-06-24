# Hamiltonian Monte Carlo for Bayesian Logistic Regression

A from-scratch implementation of Hamiltonian Monte Carlo (HMC) applied to Bayesian
binary classification on the **Breast Cancer Wisconsin** dataset (UCI / sklearn).

The project demonstrates how classical Hamiltonian mechanics — conservation laws,
symplectic geometry, Liouville's theorem — translates directly into a rigorous
statistical sampling algorithm.

---

## Project structure

```
.
├── src/
│   ├── hamiltonian.py   # U(β), ∇U(β), K(p) — the statistical Hamiltonian
│   ├── integrator.py    # Leapfrog (Störmer-Verlet) symplectic integrator
│   ├── sampler.py       # HMC chain with Metropolis-Hastings correction
│   └── diagnostics.py   # ESS, R̂, posterior summary table
├── notebooks/
│   └── 01_hmc_breast_cancer.ipynb   # End-to-end walkthrough (10 sections)
├── MEDIUM_ARTICLE.md    # Draft article for publication on Medium
├── requirements.txt
└── LICENSE
```

---

## The core idea

| Physics | Statistics |
|---|---|
| Position $q$ | Model parameters $\beta$ |
| Momentum $p$ | Auxiliary variable, refreshed each step |
| Potential energy $U(q) = -\log \pi(q \mid \text{data})$ | Negative log-posterior |
| Kinetic energy $K(p) = \frac{1}{2} p^\top M^{-1} p$ | Gaussian momentum |
| Hamilton's equations | Trajectory through parameter space |
| Liouville's theorem | Phase-space volume preserved → exact Metropolis |

The leapfrog integrator is **symplectic**: it exactly preserves a modified
Hamiltonian $\tilde{H} = H + O(\varepsilon^2)$, which is what makes the
Metropolis acceptance step exactly correct rather than approximate.

---

## Notebook contents

| Section | What it covers |
|---|---|
| 1 | Setup and imports |
| 2 | Data: feature selection, standardisation, train/test split |
| 3 | The statistical Hamiltonian + analytic gradient check |
| 4 | Phase space warm-up: leapfrog on a 2D Gaussian |
| 5 | Full HMC run on breast cancer |
| 6 | MCMC diagnostics: trace plots, ESS, R̂, energy conservation |
| 7 | Posterior predictive: calibration curve + per-sample uncertainty |
| 8 | Comparison with MLE (sklearn) |
| 9 | Joint posterior pair plot |
| 10 | Summary and natural extensions |

---

## Quick start

```bash
# Clone
git clone <repo-url>
cd mathematical-physics-hamiltonians

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/01_hmc_breast_cancer.ipynb
```

---

## Key results (typical run)

| Metric | HMC Bayesian | MLE (sklearn) |
|---|---|---|
| Accuracy | ~0.974 | ~0.974 |
| ROC-AUC | ~0.997 | ~0.996 |
| Brier score | ~0.027 | ~0.034 |

HMC produces a calibrated predictive distribution and full posterior uncertainty
quantification — the Brier score improvement reflects better-calibrated probabilities.

---

## Mathematical references

- Neal, R. M. (2011). *MCMC Using Hamiltonian Dynamics.* In Handbook of Markov
  Chain Monte Carlo. Chapman & Hall / CRC.
- Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. CRC Press.
- Betancourt, M. (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo.*
  arXiv:1701.02434.
- Leimkuhler, B. & Reich, S. (2004). *Simulating Hamiltonian Dynamics.*
  Cambridge University Press.

---

## License

MIT © 2026 Maria Oros
