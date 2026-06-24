"""
Hamiltonian components for Bayesian logistic regression.

H(q, p) = U(q) + K(p)

U(q) = -log p(q | X, y)   potential energy  (negative log-posterior)
K(p) = p^T M^{-1} p / 2   kinetic energy    (Gaussian momentum)
"""

import numpy as np


def _sigmoid(z):
    # Numerically stable sigmoid: avoids overflow for large |z|
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def log_prior(beta, sigma_prior: float = 1.0) -> float:
    """log N(beta; 0, sigma_prior^2 I)."""
    return -0.5 * np.dot(beta, beta) / sigma_prior**2


def log_likelihood(beta, X: np.ndarray, y: np.ndarray) -> float:
    """Bernoulli log-likelihood with logistic link.

    log p(y | X, beta) = sum_i [ y_i * x_i^T beta - log(1 + exp(x_i^T beta)) ]
    """
    logits = X @ beta
    return np.sum(y * logits - np.logaddexp(0.0, logits))


def potential_energy(beta, X: np.ndarray, y: np.ndarray, sigma_prior: float = 1.0) -> float:
    """U(beta) = -log p(beta | X, y)."""
    return -(log_likelihood(beta, X, y) + log_prior(beta, sigma_prior))


def grad_potential(beta, X: np.ndarray, y: np.ndarray, sigma_prior: float = 1.0) -> np.ndarray:
    """Analytic gradient of U with respect to beta.

    dU/dbeta = X^T (sigma(X beta) - y) + beta / sigma_prior^2
    """
    probs = _sigmoid(X @ beta)
    return X.T @ (probs - y) + beta / sigma_prior**2


def kinetic_energy(p: np.ndarray, M_inv: np.ndarray | None = None) -> float:
    """K(p) = p^T M^{-1} p / 2.  Defaults to M = I (unit mass matrix)."""
    if M_inv is None:
        return 0.5 * np.dot(p, p)
    return 0.5 * p @ M_inv @ p


def hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sigma_prior: float = 1.0,
    M_inv: np.ndarray | None = None,
) -> float:
    """Total energy H = U(q) + K(p)."""
    return potential_energy(q, X, y, sigma_prior) + kinetic_energy(p, M_inv)
