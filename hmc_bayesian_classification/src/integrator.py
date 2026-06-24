"""
Leapfrog (Störmer-Verlet) symplectic integrator.

The leapfrog scheme is symplectic: it exactly preserves a modified
Hamiltonian H_eps(q,p) = H(q,p) + O(eps^2), which guarantees the
Metropolis acceptance step has the correct stationary distribution.

Update equations:
    p_{t+eps/2} = p_t     - (eps/2) * grad_U(q_t)
    q_{t+eps}   = q_t     + eps * M^{-1} p_{t+eps/2}    (repeated L times)
    p_{t+eps}   = p_{t+eps/2} - (eps/2) * grad_U(q_{t+eps})
"""

from typing import Callable
import numpy as np


def leapfrog(
    q: np.ndarray,
    p: np.ndarray,
    grad_U: Callable[[np.ndarray], np.ndarray],
    epsilon: float,
    L: int,
    M_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run L leapfrog steps.

    Returns
    -------
    q_new, p_new : final position and momentum
    q_traj, p_traj : (L+1, d) arrays of the full trajectory (including start)
    """
    if M_inv is None:
        M_inv = np.eye(len(q))

    q = q.copy()
    p = p.copy()

    q_traj = [q.copy()]
    p_traj = [p.copy()]

    # Initial half-step for momentum
    p -= 0.5 * epsilon * grad_U(q)

    for _ in range(L - 1):
        q += epsilon * (M_inv @ p)
        p -= epsilon * grad_U(q)
        q_traj.append(q.copy())
        p_traj.append(p.copy())

    # Final full-step for position, then half-step for momentum
    q += epsilon * (M_inv @ p)
    p -= 0.5 * epsilon * grad_U(q)

    q_traj.append(q.copy())
    p_traj.append(p.copy())

    return q, p, np.array(q_traj), np.array(p_traj)
