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
N_train_val = 1400
X = X[:N_train_val,:]
Y = Y[:N_train_val,:]

# ───────────────────────────── TRAIN / VAL SPLIT ────────────────────────
X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.1, random_state=0)

# ────────────────────────────── NORMALISATION ───────────────────────────
EPS = 1e-6
X_mean, X_std = X_tr.mean(0), X_tr.std(0) + EPS
Y_mean, Y_std = Y_tr.mean(0), Y_tr.std(0) + EPS

X_tr_norm = (X_tr - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std

# ────────────────────────────── PROBLEMDATA ─────────────────────────────
problem = ProblemData(
    Q=jnp.array(Q), A=jnp.array(A), G=jnp.array(G), p=jnp.array(p), h=jnp.array(h),
    X_train_norm=jnp.array(X_tr_norm),
    X_train=jnp.array(X_tr),
    X_val_norm=jnp.array(X_val_norm),
    X_val=jnp.array(X_val),
    # Y_train is optional for unsupervised but kept for normalisation
    Y_train=jnp.array(Y_tr),
    Y_val=jnp.array(Y_val),
    Y_mean=jnp.array(Y_mean),
    Y_std=jnp.array(Y_std),
    batch_size=1000,
)
problem.init_meta(hidden_width=240, num_hidden_layers=1)

# ───────────────────────────── LOGGER & EARLY STOP ──────────────────────
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('dc3-unsup-train')

stop_check = PatienceThresholdStoppingCriteria(log, threshold=1e-8, patience=3)

# ─────────────────────────────── TRAINING ───────────────────────────────
log.info('  starting *unsupervised* training …')

um.run_unsupervised(
    problem_data=problem,
    log=log,
    initial_learning_rate=1e-3,
    decay_rate=1e-4,
    max_training_time=300.0,
    max_epochs=2000,
    validate_every=10,
    vi_parameters=None,
    stop_check=stop_check,
    rng_key=key,
)

log.info('  training finished')

best_params = stop_check.vi_parameters

# ───────────────────────────── PREDICTION ────────────────────────────────
key, subkey = jax.random.split(key)

Y_pred, Y_pred_std = um.predict_unsupervised(
    rng_key=subkey,
    params=best_params,
    X_norm=jnp.array(X_val_norm),
    X_raw=jnp.array(X_val),
    problem_data=problem,
    num_samples=500,
)

# ───────────────────────────── DIAGNOSTICS ───────────────────────────────
r_eq   = equality_residuals(problem.X_val, Y_pred, problem)
r_ineq = inequality_residuals(problem.X_val, Y_pred, problem)
obj    = 0.5 * jnp.sum(Y_pred * (problem.Q @ Y_pred.T).T, axis=1) + \
         jnp.sum(problem.p * jnp.sin(Y_pred), axis=1)

print('\nValidation summary (unsupervised)')
print(f'  max ‖Ay−x‖₂   : {jnp.linalg.norm(r_eq, axis=1).max():.3E}')
print(f'  max max(0,Gy−h): {r_ineq.max(1).max():.3E}')
print(f'  min objective  : {obj.min()}')

# ─────────────────────── TRUE VALUES ON VALIDATION SET ───────────────────
r_eq_true   = equality_residuals(problem.X_val, problem.Y_val, problem)
r_ineq_true = inequality_residuals(problem.X_val, problem.Y_val, problem)
obj_true    = 0.5 * jnp.sum(problem.Y_val * (problem.Q @ problem.Y_val.T).T, axis=1) + \
             jnp.sum(problem.p * jnp.sin(problem.Y_val), axis=1)

print('\nGround‑truth (validation targets)')
print(f'  max ‖Ay−x‖₂   : {jnp.linalg.norm(r_eq_true, axis=1).max():.3E}')
print(f'  max max(0,Gy−h): {r_ineq_true.max(1).max():.3E}')
print(f'  min objective  : {obj_true.min()}')

print(f'MSE :{jnp.linalg.norm(Y_val - Y_pred, axis = 1).max()/ jnp.linalg.norm(Y_val, axis = 1).max()}')

# ───────────────────────────── SAVE LOG (optional) ───────────────────────
if input('\nSave training log to "train_log_unsup.npy"? [y/N] → ').lower().startswith('y'):
    np.save('train_log_unsup.npy', stop_check.history)
    print('log saved.')
