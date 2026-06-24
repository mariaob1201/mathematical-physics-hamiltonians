"""
Hamiltonian Monte Carlo sampler.

Each HMC transition:
1. Sample auxiliary momentum p ~ N(0, M)
2. Simulate L leapfrog steps from (q, p) to get proposal (q*, p*)
3. Accept/reject via Metropolis on exp(-H)

The Metropolis step corrects for numerical integration error from the
leapfrog scheme, preserving exact detailed balance.
"""

from __future__ import annotations
import numpy as np
from .integrator import leapfrog
from .hamiltonian import hamiltonian, grad_potential


def hmc_step(
    q: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    L: int,
    sigma_prior: float = 1.0,
    M: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """Single HMC transition.

    Returns
    -------
    q_next   : accepted or rejected position
    accepted : whether the proposal was accepted
    q_traj   : leapfrog position trajectory (for diagnostics)
    p_traj   : leapfrog momentum trajectory
    """
    d = len(q)
    M = np.eye(d) if M is None else M
    M_inv = np.linalg.inv(M)

    # Refresh momentum
    p = np.random.multivariate_normal(np.zeros(d), M)

    H_current = hamiltonian(q, p, X, y, sigma_prior, M_inv)

    grad_U = lambda beta: grad_potential(beta, X, y, sigma_prior)
    q_prop, p_prop, q_traj, p_traj = leapfrog(q, p, grad_U, epsilon, L, M_inv)

    H_proposed = hamiltonian(q_prop, p_prop, X, y, sigma_prior, M_inv)

    # Metropolis acceptance: alpha = min(1, exp(H_current - H_proposed))
    log_alpha = H_current - H_proposed
    accepted = np.log(np.random.uniform()) < log_alpha

    q_next = q_prop if accepted else q
    return q_next, bool(accepted), q_traj, p_traj


def hmc_sample(
    q_init: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    epsilon: float,
    L: int,
    sigma_prior: float = 1.0,
    M: np.ndarray | None = None,
    n_warmup: int = 500,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Run a single HMC chain.

    Parameters
    ----------
    q_init    : starting position in parameter space
    X, y      : design matrix and binary labels
    n_samples : number of post-warmup draws
    epsilon   : leapfrog step size
    L         : number of leapfrog steps per transition
    sigma_prior : prior standard deviation (isotropic Gaussian)
    M         : mass matrix (defaults to identity)
    n_warmup  : number of discarded warmup transitions
    seed      : random seed for reproducibility
    verbose   : print progress

    Returns
    -------
    samples : (n_samples, d) posterior draws
    info    : dict with acceptance_rate, warmup_acceptance_rate,
              last_q_traj, last_p_traj
    """
    if seed is not None:
        np.random.seed(seed)

    q = q_init.copy()
    d = len(q)
    M = np.eye(d) if M is None else M

    # --- Warmup ---
    n_acc_warmup = 0
    if verbose:
        print(f"  Warmup  ({n_warmup} steps, eps={epsilon}, L={L}) ...")
    for _ in range(n_warmup):
        q, accepted, _, _ = hmc_step(q, X, y, epsilon, L, sigma_prior, M)
        n_acc_warmup += int(accepted)
    warmup_rate = n_acc_warmup / n_warmup
    if verbose:
        print(f"  Warmup acceptance rate : {warmup_rate:.1%}")

    # --- Sampling ---
    samples = np.zeros((n_samples, d))
    n_acc = 0
    last_q_traj = last_p_traj = None

    if verbose:
        print(f"  Sampling ({n_samples} draws) ...")
    for i in range(n_samples):
        q, accepted, q_traj, p_traj = hmc_step(q, X, y, epsilon, L, sigma_prior, M)
        samples[i] = q
        n_acc += int(accepted)
        if i == n_samples - 1:
            last_q_traj, last_p_traj = q_traj, p_traj

    sampling_rate = n_acc / n_samples
    if verbose:
        print(f"  Sampling acceptance rate: {sampling_rate:.1%}")

    info = {
        "acceptance_rate": sampling_rate,
        "warmup_acceptance_rate": warmup_rate,
        "last_q_traj": last_q_traj,
        "last_p_traj": last_p_traj,
    }
    return samples, info
