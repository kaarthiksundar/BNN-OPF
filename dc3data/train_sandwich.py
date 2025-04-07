"""train_dc3_unsupervised_bnn.p
y
================================
End‑to‑end **unsupervised** training script for the quadratic–plus–sine
problem using the DC‑3 Bayesian‑NN codebase.

Differences from the supervised version
---------------------------------------
* Calls ``unsupervisedmodel.run_unsupervised`` instead of the supervised loop.
* Uses the helper ``predict_unsupervised`` that queries the likelihood site
  "Y_y" (no deterministic site needed).
* No target labels are supplied during training; we only use ``Y_val`` to
  evaluate residuals and objective after training.
"""
from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

import dc3unsupervisedmodel as um            # model, guide, training loop & predict helper
import dc3supervisedmodel as sm
from dc3classes import ProblemData
from dc3feasibility import equality_residuals, inequality_residuals
from stopping import PatienceThresholdStoppingCriteria

# ─────────────────────────────── RNG KEY ────────────────────────────────
key = jax.random.PRNGKey(0)

# ────────────────────────────── LOAD DATA ───────────────────────────────
# Suppose X.shape == (N, m)   and   Y.shape == (N, n)
#filename = "random_nonconvex_dataset_var20_ineq5_eq10_ex5000.npz"
filename = 'random_nonconvex_dataset_var100_ineq50_eq50_ex10000.npz'
#filename = 'random_nonconvex_dataset_var150_ineq50_eq50_ex5000.npz'
data = np.load(filename, allow_pickle=False)
G, Q, A, h, p, X, Y = (data[k] for k in ('G','Q','A','h','p','X','Y'))

N_unsup = 2048
N_sup = 512
N_test = 100
N_tot = N_unsup + N_sup + N_test


idx1 = N_sup
idx2 = idx1 + N_unsup
idx3 = idx2 + N_val

X_sup,  Y_sup  = X[:idx1,],         Y[:idx1]
X_unsup, Y_unsup = X[idx1:idx2],    Y[idx1:idx2]
X_val,  Y_val  = X[idx2:idx3],     Y[idx2:idx3]
X_test, Y_test = X[idx3:idx3+N_test], Y[idx3:idx3+N_test]



# ───────────────────────────── TRAIN / VAL SPLIT ────────────────────────

# ────────────────────────────── NORMALISATION ───────────────────────────
EPS = 1e-6
EPS = 1e-6
X_mean, X_std = X[:idx2].mean(0), X[:idx2].std(0) + EPS
Y_mean, Y_std = Y[:idx2].mean(0), Y[:idx2].std(0) + EPS



X_sup_norm = (X_sup - X_mean) / X_std
X_unsup_norm = (X_unsup - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std
problem_unsup = ProblemData(
    Q=jnp.array(Q), A=jnp.array(A), G=jnp.array(G), p=jnp.array(p), h=jnp.array(h),
    X_train_norm=jnp.array(X_unsup_norm),
    X_train=jnp.array(X_unsup),
    X_val_norm=jnp.array(X_val_norm),
    X_val=jnp.array(X_val),
    # Y_train is optional for unsupervised but kept for normalisation
    Y_train=jnp.array(Y_unsup),
    Y_val=jnp.array(Y_val),
    Y_mean=jnp.array(Y_mean),
    Y_std=jnp.array(Y_std),
    batch_size=1000,
)
problem_unsup.init_meta(hidden_width=240, num_hidden_layers=1)
problem_sup = ProblemData(
    Q=jnp.array(Q), A=jnp.array(A), G=jnp.array(G), p=jnp.array(p), h=jnp.array(h),
    X_train_norm=jnp.array(X_sup_norm),
    X_train=jnp.array(X_sup),
    X_val_norm=jnp.array(X_val_norm),
    X_val=jnp.array(X_val),
    # Y_train is optional for unsupervised but kept for normalisation
    Y_train=jnp.array(Y_sup),
    Y_val=jnp.array(Y_val),
    Y_mean=jnp.array(Y_mean),
    Y_std=jnp.array(Y_std),
    batch_size=1000,
)


problem_sup.init_meta(hidden_width=240, num_hidden_layers=1)

