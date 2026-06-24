"""
Nonlinear pendulum data generator.

Hamiltonian (natural units: m = g = L = 1):
    H(q, p) = p²/2 + (1 − cos q)

Equations of motion:
    dq/dt =  ∂H/∂p = p
    dp/dt = −∂H/∂q = −sin(q)

The pendulum is integrable and has a conserved energy H(q,p) = const,
making it ideal for demonstrating that HNN rollouts conserve energy
while a standard MLP does not.
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp


def pendulum_H(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True Hamiltonian: H = p²/2 + (1 − cos q)."""
    return 0.5 * p ** 2 + (1 - np.cos(q))


def pendulum_derivatives(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True time derivatives: dq/dt = p,  dp/dt = −sin(q)."""
    return p, -np.sin(q)


def _ode(t, y):
    return [y[1], -np.sin(y[0])]


def generate_data(
    n_traj: int = 40,
    n_points: int = 50,
    t_span: tuple[float, float] = (0.0, 8.0),
    noise_std: float = 0.01,
    seed: int = 42,
    q_range: tuple[float, float] = (-2.5, 2.5),
    p_range: tuple[float, float] = (-2.0, 2.0),
) -> dict[str, np.ndarray]:
    """Generate noisy pendulum trajectory data.

    Samples n_traj initial conditions uniformly in (q_range, p_range),
    integrates each with scipy solve_ivp, then samples n_points from each.

    Returns a dict with keys:
        X       : (N, 2) array of (q, p) phase-space points
        dX      : (N, 2) array of (dq_dt, dp_dt) derivatives
        H       : (N,)  true Hamiltonian values
        X_noisy : (N, 2) with added Gaussian noise (for training)
    """
    rng = np.random.RandomState(seed)
    t_eval = np.linspace(*t_span, n_points)

    q_list, p_list, dq_list, dp_list, H_list = [], [], [], [], []

    for _ in range(n_traj):
        q0 = rng.uniform(*q_range)
        p0 = rng.uniform(*p_range)

        # Skip trajectories with energy above separatrix (unbound rotation)
        if pendulum_H(q0, p0) > 1.98:
            continue

        sol = solve_ivp(_ode, t_span, [q0, p0], t_eval=t_eval,
                        method="DOP853", rtol=1e-9, atol=1e-9)
        if not sol.success:
            continue

        q, p = sol.y[0], sol.y[1]
        dq, dp = pendulum_derivatives(q, p)

        q_list.append(q)
        p_list.append(p)
        dq_list.append(dq)
        dp_list.append(dp)
        H_list.append(pendulum_H(q, p))

    q_all  = np.concatenate(q_list)
    p_all  = np.concatenate(p_list)
    dq_all = np.concatenate(dq_list)
    dp_all = np.concatenate(dp_list)
    H_all  = np.concatenate(H_list)

    X  = np.column_stack([q_all, p_all])
    dX = np.column_stack([dq_all, dp_all])

    noise = rng.randn(*X.shape) * noise_std

    return {
        "X":       X,
        "dX":      dX,
        "H":       H_all,
        "X_noisy": X + noise,
    }


def rollout(
    q0: float,
    p0: float,
    n_steps: int,
    dt: float,
    dyn_fn,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a dynamical system from (q0, p0) for n_steps.

    Parameters
    ----------
    dyn_fn : callable(q, p) → (dq_dt, dp_dt)
    method : 'rk4' or 'euler'

    Returns
    -------
    qs, ps : (n_steps+1,) arrays of trajectory
    """
    qs = np.zeros(n_steps + 1)
    ps = np.zeros(n_steps + 1)
    qs[0], ps[0] = q0, p0

    for i in range(n_steps):
        q, p = qs[i], ps[i]
        if method == "rk4":
            k1q, k1p = dyn_fn(q, p)
            k2q, k2p = dyn_fn(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
            k3q, k3p = dyn_fn(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
            k4q, k4p = dyn_fn(q + dt * k3q, p + dt * k3p)
            qs[i+1] = q + dt * (k1q + 2*k2q + 2*k3q + k4q) / 6
            ps[i+1] = p + dt * (k1p + 2*k2p + 2*k3p + k4p) / 6
        else:
            dq, dp = dyn_fn(q, p)
            qs[i+1] = q + dt * dq
            ps[i+1] = p + dt * dp

    return qs, ps
