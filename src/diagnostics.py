"""
MCMC diagnostics: ESS, R-hat, and summary statistics.
"""

from __future__ import annotations
import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Estimate normalized autocorrelation function up to max_lag."""
    n = len(x)
    x = x - x.mean()
    variance = np.var(x)
    if variance == 0:
        return np.zeros(max_lag + 1)
    # Full correlation via FFT for efficiency
    full = np.correlate(x, x, mode="full")
    ac = full[n - 1 :] / (variance * n)
    return ac[: max_lag + 1]


def effective_sample_size(chain: np.ndarray) -> float:
    """ESS via the initial positive sequence estimator (Geyer 1992).

    ESS = N / (1 + 2 * sum_{k>=1} rho_k)
    where the sum stops at the first negative autocorrelation.
    """
    n = len(chain)
    ac = autocorrelation(chain, max_lag=n // 2)
    rho_sum = 0.0
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            break
        rho_sum += ac[k]
    return n / max(1.0, 1.0 + 2.0 * rho_sum)


def r_hat(chains: list[np.ndarray]) -> float:
    """Gelman-Rubin potential scale reduction factor.

    Values close to 1.0 indicate convergence.  Values > 1.1 suggest
    that more sampling is needed.
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([c.mean() for c in chains])
    grand_mean = chain_means.mean()
    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains])
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def summary(samples: np.ndarray, param_names: list[str] | None = None) -> dict:
    """Return mean, std, 2.5%, 50%, 97.5% and ESS for each parameter."""
    n_params = samples.shape[1]
    if param_names is None:
        param_names = [f"beta_{i}" for i in range(n_params)]

    rows = []
    for i, name in enumerate(param_names):
        chain = samples[:, i]
        rows.append(
            {
                "param": name,
                "mean": chain.mean(),
                "std": chain.std(),
                "2.5%": np.percentile(chain, 2.5),
                "50%": np.percentile(chain, 50),
                "97.5%": np.percentile(chain, 97.5),
                "ESS": effective_sample_size(chain),
            }
        )
    return rows
