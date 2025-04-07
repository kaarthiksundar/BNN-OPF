"""Train a *supervised* Bayesian neural network for the quadratic–plus–sine
problem using the DC‑3 flavour of the training loop defined in
`dc3supervisedmodel.py`.

The script is intentionally minimal:
    1. loads data (replace the numpy loads with your own I/O)
    2. splits into train / validation
    3. normalises X and Y
    4. builds a `ProblemData` container
    5. calls `run_supervised` from `dc3supervisedmodel`
    6. prints constraint residuals and objective on the validation set.

Anything related to *logging* is handled inside `dc3supervisedmodel`; the
function returns a `train_log` object that you can save if you want.
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import Predictive
from sklearn.model_selection import train_test_split
import dc3supervisedmodel as dcs 
from dc3classes import ProblemData
from dc3feasibility import equality_residuals, inequality_residuals

# ─── import the DC‑3 training / prediction helpers ─────────────────────────
from dc3supervisedmodel import run_supervised, predict_supervised  # adapt names if different
from stopping import PatienceThresholdStoppingCriteria
# ────────────────────────────── 0.  RNG key ────────────────────────────────
key = jax.random.PRNGKey(0)

# ────────────────────────────── 1.  load data ──────────────────────────────
# Suppose X.shape == (N, m)   and   Y.shape == (N, n)
#filename = "random_nonconvex_dataset_var20_ineq5_eq10_ex5000.npz"
filename = 'random_nonconvex_dataset_var100_ineq50_eq50_ex10000.npz'
#filename = 'random_nonconvex_dataset_var150_ineq50_eq50_ex5000.npz'
data = np.load(filename, allow_pickle=False)
G, Q, A, h, p, X, Y = (data[k] for k in ('G','Q','A','h','p','X','Y'))
N_train_val = 500
X = X[:N_train_val,:]
Y = Y[:N_train_val,:]


# ────────────────────────────── 2.  split ─────────────────────────────────
X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=0.1, random_state=0)

# ────────────────────────────── 3.  normalisation ─────────────────────────
EPS = 1e-6
X_mean, X_std = X_tr.mean(0), X_tr.std(0) + EPS
Y_mean, Y_std = Y_tr.mean(0), Y_tr.std(0) + EPS

X_tr_norm = (X_tr - X_mean) / X_std
X_val_norm = (X_val - X_mean) / X_std

# ────────────────────────────── 4.  ProblemData ───────────────────────────
problem = ProblemData(
    Q=jnp.array(Q), A=jnp.array(A), G=jnp.array(G), h=jnp.array(h), p=jnp.array(p),
    X_train_norm=X_tr_norm,  # placeholders for API compatibility
    X_train=X_tr,
    X_val_norm=X_val_norm,
    X_val=X_val,
    Y_train =  Y_tr,
    Y_val = Y_val,
    Y_mean=jnp.array(Y_mean),
    Y_std=jnp.array(Y_std),
    batch_size=500
)
problem.init_meta(hidden_width=240, num_hidden_layers=1)


# 0. logger -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("dc3-train")

# 1. early‑stopping ----------------------------------------------------------
stop_check = PatienceThresholdStoppingCriteria(log,
                                               threshold=1e-8,
                                               patience=5)

# 2. train -------------------------------------------------------------------
run_supervised(problem,           # ProblemData
               log,               # logger
               initial_learning_rate=1e-2,
               decay_rate=5e-3,
               max_training_time=300.0,
               max_epochs=2000,
               validate_every=20,
               vi_parameters=None,
               stop_check=stop_check,
               rng_key=key)

# 3. build a Predictive object with the *best* parameters --------------------

best_params = stop_check.vi_parameters            # ← stored by early‑stopping
key, subkey = jax.random.split(key)

# helper handles argument order expected by the model
predictive = Predictive(
    model=dcs.supervised_testing_model if hasattr(dcs, "supervised_testing_model") else dcs.supervised_model,
    guide=dcs.supervised_guide,
    params=best_params,              # overrides sample sites with MAP/VI params
    num_samples=500,
    return_sites=["Y_y"],
)

samples = predictive(
    subkey,
    X_val_norm,                     # X_norm
    X_val,                          # X_raw
    Y=None,                         # no ground‑truth
    problem_data=problem,
    vi_parameters=best_params,      # passed so model has access if needed
)

Y_pred_norm = samples["Y_y"].mean(axis=0)
Y_pred      = Y_pred_norm * problem.Y_std + problem.Y_mean

# ────────────────────────────── 7.  diagnostics ───────────────────────────
# equality:  Ay - x
r_eq   = equality_residuals(jnp.array(X_val), Y_pred, problem)
# inequality:  max(0, Gy - h)
r_ineq = inequality_residuals(jnp.array(X_val), Y_pred, problem)

# objective  ½ yᵀQy + pᵀ sin(y)
obj = 0.5 * jnp.sum(Y_pred * (problem.Q @ Y_pred.T).T, axis=1) + jnp.sum(problem.p * jnp.sin(Y_pred), axis=1)

print("Validation summary:")
print("  max ‖Ay−x‖₂ :", jnp.linalg.norm(r_eq, axis=1).max())
print("  max max(0, Gy−h) :", r_ineq.max(1).max())
print("  min objective     :", obj.min())

# ─────────────────── TRUE VALUES ON VALIDATION SET ────────────────────
r_eq_true   = equality_residuals(problem.X_val, problem.Y_val, problem)
r_ineq_true = inequality_residuals(problem.X_val, problem.Y_val, problem)
obj_true    = (
    0.5 * jnp.sum(problem.Y_val * (problem.Q @ problem.Y_val.T).T, axis=1)
    + jnp.sum(problem.p * jnp.sin(problem.Y_val), axis=1)
)

print('\nGround‑truth (validation targets)')
print('  max ‖Ay−x‖₂   :', jnp.linalg.norm(r_eq_true, axis=1).max())
print('  max max(0,Gy−h):', r_ineq_true.max(1).max())
print('  min objective  :', obj_true.min())

print(f'MSE :{jnp.linalg.norm(Y_val - Y_pred, axis = 1).max()/ jnp.linalg.norm(Y_val, axis = 1).max()}')

# optional: persist training log
after = input("Save training log to 'train_log.npy'? [y/N] → ")
if after.lower().startswith("y"):
    np.save("train_log.npy", train_log)
    print("  log saved.")
