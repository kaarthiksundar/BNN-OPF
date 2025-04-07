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
import jax.random as random
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
from bnncommon import *
import time

# ─────────────────────────────── RNG KEY ────────────────────────────────
key = jax.random.PRNGKey(0)

# ────────────────────────────── LOAD DATA ───────────────────────────────
# Suppose X.shape == (N, m)   and   Y.shape == (N, n)
#filename = "random_nonconvex_dataset_var20_ineq5_eq10_ex5000.npz"
filename = 'random_nonconvex_dataset_var100_ineq50_eq50_ex10000.npz'
#filename = 'random_nonconvex_dataset_var150_ineq50_eq50_ex5000.npz'
data = np.load(filename, allow_pickle=False)
G, Q, A, h, p, X, Y = (data[k] for k in ('G','Q','A','h','p','X','Y'))


N_unsup = 4096
N_sup = 512
N_val = 100
N_test = 100
N_tot = N_unsup + N_sup + N_test + N_val
assert N_tot <= X.shape[0]


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
    Y_train=jnp.array(Y_unsup),
    Y_val=jnp.array(Y_val),
    Y_mean=jnp.array(Y_mean),
    Y_std=jnp.array(Y_std),
    batch_size=256,
)
problem_unsup.init_meta(hidden_width=240, num_hidden_layers=1)
problem = ProblemData(
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
    batch_size=256,
)


problem.init_meta(hidden_width=240, num_hidden_layers=1)
config = {}
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger('dc3-sandwich-train')



initial_learning_rate = config.get("initial_learning_rate", 1e-3)
decay_rate = config.get("decay_rate", 1e-4) 
sandwich_rounds = config.get("sandwich_rounds", 10) 
max_training_time_per_round = config.get("max_training_time_per_round", 200.0)
max_training_time = config.get("max_training_time", 1000.0)
max_epochs = config.get("max_epochs", 2000) 
early_stopping_trigger_supervised = config.get("early_stopping_trigger_supervised", 25) 
early_stopping_trigger_unsupervised = config.get("early_stopping_trigger_unsupervised", 30)
patience_supervised = config.get("patience_supervised", 3)
patience_unsupervised = config.get("patience_unsupervised", 5)

# create early stopping for both the supervised and unsupervised runs
supervised_early_stopper = PatienceThresholdStoppingCriteria(
    log, patience = patience_supervised)
unsupervised_early_stopper = PatienceThresholdStoppingCriteria(
    log, patience = patience_unsupervised)
sandwiched_early_stopper = PatienceThresholdStoppingCriteria(
    log, patience = 3
)

max_time_supervised = 0.1 * max_training_time_per_round 
max_time_unsupervised = 0.9 * max_training_time_per_round
supervised_params = []
unsupervised_params = []
vi_parameters = None 
model_params = get_model_params(problem)
remaining_time = max_training_time
start_time = time.time() 
rng_key = random.PRNGKey(0)
 

for round in range(sandwich_rounds): 
    log.info(f'round number: {round + 1}')
    sm.run_supervised(
            problem,           # ProblemData
            log,                  # logger
            initial_learning_rate = initial_learning_rate/(round + 1), 
            decay_rate = decay_rate/(round + 1), 
            max_training_time = min(remaining_time, max_time_supervised), 
            max_epochs =30, 
            validate_every = early_stopping_trigger_supervised, 
            vi_parameters = vi_parameters, 
            stop_check = supervised_early_stopper, 
            rng_key = rng_key
        )
    validation_loss = supervised_early_stopper.best_loss
    log.info(f'supervised validation loss: {validation_loss}')
    vi_parameters = supervised_early_stopper.vi_parameters
    supervised_params.append(vi_parameters)
    supervised_early_stopper.reset_wait() 
    sandwiched_early_stopper.on_epoch_end(
        round + 1, 
        supervised_early_stopper.best_loss, 
        supervised_early_stopper.vi_parameters)
    if sandwiched_early_stopper.stop_training == True:
        log.info(f'Stopping criteria for sandwiched algorithm passed at {round + 1}, breaking')
        break
    # check overall time
    elapsed = time.time() - start_time
    remaining_time = max_training_time - elapsed
    if time.time() - start_time > max_training_time:
        log.info(f'Maximum training time exceeded at supervised round {round + 1}')
        break
    log.info('  starting *unsupervised* training …')

    um.run_unsupervised(
        problem_data=problem_unsup,
        log=log,
        initial_learning_rate = initial_learning_rate/(round + 1),
        decay_rate = decay_rate/(round + 1), 
        max_training_time = min(remaining_time, max_time_unsupervised), 
        max_epochs = max_epochs, 
        validate_every = early_stopping_trigger_unsupervised, 
        vi_parameters=vi_parameters,
        stop_check = unsupervised_early_stopper, 
        rng_key=key,
    )

    validation_loss = unsupervised_early_stopper.best_loss 
    log.info(f'unsupervised validation loss: {validation_loss}')
    vi_parameters = unsupervised_early_stopper.vi_parameters 
    for name in model_params['output_block_dim'].keys(): 
        mean_key = f'l_std_{name}_mean'
        std_key = f'l_std_{name}_std'
        vi_parameters[mean_key] = supervised_params[-1][mean_key]
        vi_parameters[std_key] = supervised_params[-1][std_key]
    unsupervised_params.append(vi_parameters)
    unsupervised_early_stopper.reset_wait()
 
    validate_model(key,
                   X_val_norm, X_val, Y_val,
                   problem, vi_parameters,
                   sm.supervised_testing_model,
                   sm.supervised_guide)
        
        # check overall time
    elapsed = time.time() - start_time
    remaining_time = max_training_time - elapsed
    if time.time() - start_time > max_training_time:
        log.info(f'Maximum training time exceeded at unsupervised round {round + 1}')
        break








