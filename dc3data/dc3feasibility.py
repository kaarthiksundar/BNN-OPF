# constraints.py
import jax.numpy as jnp
from dc3classes import ProblemData

def equality_residuals(X: jnp.ndarray, Y: jnp.ndarray, pdata: ProblemData) -> jnp.ndarray:
    """
    Returns (batch, m) array of equality residuals  r_eq = A y - x .
    """
    return jnp.einsum('mn,bn->bm', pdata.A, Y) - X


def inequality_residuals(X: jnp.ndarray, Y: jnp.ndarray, pdata: ProblemData) -> jnp.ndarray:
    """
    Returns (batch, k) array of *positive* inequality violations  r_ineq = max(0, G y - h) .
    """
    raw = jnp.einsum('kn,bn->bk', pdata.G, Y) - pdata.h
    return jnp.maximum(0.0, raw)


def assess_feasibility(X: jnp.ndarray,
                       Y: jnp.ndarray,
                       pdata: ProblemData,
                       eq_weight: float = 1.0,
                       ineq_weight: float = 1.0) -> jnp.ndarray:
    """
    Scalar non‑negative feasibility loss per sample (shape: (batch,)).
    """
    eq   = equality_residuals(X, Y, pdata)        # (b, m)
    ineq = inequality_residuals(X, Y, pdata)      # (b, k)

    return (eq_weight  * jnp.square(eq).sum(axis=1) +
            ineq_weight * jnp.square(ineq).sum(axis=1))
