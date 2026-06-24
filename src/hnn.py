"""
Hamiltonian Neural Network (HNN) — numpy-only implementation.

Architecture: MLP [2 → hidden → hidden → 1] with tanh activations.
The network learns H(q, p) from energy labels.
Dynamics are then derived analytically:
    dq/dt =  ∂H/∂p
    dp/dt = -∂H/∂q

The input Jacobian ∂H/∂x is computed via the closed-form chain rule
through the network (no autograd needed for a fixed 2-hidden-layer MLP).

Reference: Greydanus et al. (2019) "Hamiltonian Neural Networks"
"""

from __future__ import annotations
import numpy as np


class Adam:
    """Adam optimiser (Kingma & Ba 2015) with a parameter-dict interface."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._m: dict = {}
        self._v: dict = {}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for k, g in grads.items():
            self._m.setdefault(k, np.zeros_like(g))
            self._v.setdefault(k, np.zeros_like(g))
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * g ** 2
            params[k] -= lr_t * self._m[k] / (np.sqrt(self._v[k]) + self.eps)


class MLP:
    """General 2-hidden-layer MLP [n_in → hidden → hidden → n_out].

    Used as the baseline model that learns (dq/dt, dp/dt) directly
    (set n_out=2) and as the HNN backbone (n_out=1, learning H).
    """

    def __init__(self, n_in: int = 2, n_out: int = 1,
                 hidden: int = 64, seed: int = 0):
        rng = np.random.RandomState(seed)
        s = lambda n: np.sqrt(2.0 / n)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.randn(n_in, hidden) * s(n_in),
            "b1": np.zeros(hidden),
            "W2": rng.randn(hidden, hidden) * s(hidden),
            "b2": np.zeros(hidden),
            "W3": rng.randn(hidden, n_out) * s(hidden),
            "b3": np.zeros(n_out),
        }
        self._cache: dict = {}

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, n_in) → output: (N, n_out). Caches activations."""
        p = self.params
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.tanh(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.tanh(z2)
        out = a2 @ p["W3"] + p["b3"]
        self._cache = {"x": x, "a1": a1, "z1": z1, "a2": a2, "z2": z2}
        return out

    # ── backward ─────────────────────────────────────────────────────────────
    def backward(self, d_out: np.ndarray) -> dict[str, np.ndarray]:
        """Standard backprop. d_out: gradient of loss w.r.t. network output."""
        p, c = self.params, self._cache
        N = len(c["x"])

        dW3 = c["a2"].T @ d_out / N
        db3 = d_out.mean(axis=0)
        da2 = d_out @ p["W3"].T

        dz2 = da2 * (1 - c["a2"] ** 2)
        dW2 = c["a1"].T @ dz2 / N
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ p["W2"].T

        dz1 = da1 * (1 - c["a1"] ** 2)
        dW1 = c["x"].T @ dz1 / N
        db1 = dz1.mean(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                "W3": dW3, "b3": db3}

    # ── input Jacobian ────────────────────────────────────────────────────────
    def input_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic ∂H/∂x for each sample. Only valid for scalar output.

        Chain rule through the 2-hidden-layer network:

            ∂H/∂x = W1 ⊙ d1 ⊙ (W2 ⊙ d2 ⊙ W3)

        In batch row-vector notation (all ops are batched over N samples):

            inner1 = d2 * W3[:,0]      # ∂H/∂z2
            inner2 = inner1 @ W2.T     # ∂H/∂a1  (backprop through W2)
            inner3 = d1  * inner2      # ∂H/∂z1
            dHdx   = inner3 @ W1.T     # ∂H/∂x   (backprop through W1)

        Returns dHdx: (N, n_in).
        """
        p = self.params
        self.forward(x)
        c = self._cache
        d1 = 1 - c["a1"] ** 2   # (N, hidden)
        d2 = 1 - c["a2"] ** 2   # (N, hidden)

        inner1 = d2 * p["W3"][:, 0]   # (N, hidden)
        inner2 = inner1 @ p["W2"].T    # (N, hidden)
        inner3 = d1 * inner2            # (N, hidden)
        return inner3 @ p["W1"].T       # (N, n_in)


# ── training helpers ─────────────────────────────────────────────────────────

def train_hnn(
    net: MLP,
    X: np.ndarray,
    H_true: np.ndarray,
    n_epochs: int = 3000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> list[float]:
    """Train HNN to predict H(q, p) from energy labels.

    Loss: MSE(H_pred, H_true)
    Returns list of per-epoch losses.
    """
    rng = np.random.RandomState(seed)
    opt = Adam(lr=lr)
    N = len(X)
    losses = []

    for epoch in range(1, n_epochs + 1):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = idx[start: start + batch_size]
            xb = X[batch]
            Hb = H_true[batch].reshape(-1, 1)

            H_pred = net.forward(xb)
            residual = H_pred - Hb
            loss = (residual ** 2).mean()
            epoch_loss += loss
            n_batches += 1

            d_out = 2 * residual / len(xb)
            grads = net.backward(d_out)
            opt.step(net.params, grads)

        losses.append(epoch_loss / n_batches)
        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | MSE(H) = {losses[-1]:.6f}")

    return losses


def train_baseline(
    net: MLP,
    X: np.ndarray,
    dX_true: np.ndarray,
    n_epochs: int = 3000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> list[float]:
    """Train baseline MLP to predict (dq/dt, dp/dt) directly.

    Loss: MSE on derivatives.
    Returns list of per-epoch losses.
    """
    rng = np.random.RandomState(seed)
    opt = Adam(lr=lr)
    N = len(X)
    losses = []

    for epoch in range(1, n_epochs + 1):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = idx[start: start + batch_size]
            xb = X[batch]
            dxb = dX_true[batch]

            dx_pred = net.forward(xb)
            residual = dx_pred - dxb
            loss = (residual ** 2).mean()
            epoch_loss += loss
            n_batches += 1

            d_out = 2 * residual / len(xb)
            grads = net.backward(d_out)
            opt.step(net.params, grads)

        losses.append(epoch_loss / n_batches)
        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch:4d} | MSE(dX) = {losses[-1]:.6f}")

    return losses
