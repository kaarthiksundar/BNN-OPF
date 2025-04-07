from dc3classes import ProblemData
from dc3feasibility import *
from typing import Dict
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
from collections import OrderedDict
from typing import Any, Dict 
from numpyro.infer import Predictive

def get_model_params(problem_data: ProblemData) -> Dict[str, Any]:
    """Return a dictionary of hyper‑parameters that fully specifies the
    Bayesian neural‑network architecture for the generic optimisation
    problem encapsulated by ``ProblemData``.

    Parameters
    ----------
    problem_data : ProblemData
        Container holding the matrices (Q, A, G, h, p) that define the
        optimisation task as well as normalisation statistics and any
        meta‑information about the desired network architecture.

    Returns
    -------
    Dict[str, Any]
        Keys required elsewhere in the code base (output_block_dim,
        output_block_slices, num_nodes_per_hidden_layer, output_dim,
        num_layers, weight_prior_std_multiplier, likelihood_var_prior_mean,
        likelihood_var_prior_std).
    """
    # ── basic dimensions ────────────────────────────────────────────────
    n = problem_data.Q.shape[0]        # number of y‑variables

    # ── output description ──────────────────────────────────────────────
    output_dim = OrderedDict([('y', n)])
    output_slices = OrderedDict([('y', slice(0, n))])

    # ── architecture hyper‑parameters (with sensible defaults) ─────────
    if problem_data.meta is None:
        hidden_width = 128
        num_layers   = 2
    else:
        hidden_width = problem_data.meta.get('hidden_width', 128)
        num_layers   = problem_data.meta.get('num_hidden_layers', 2)

    # ── assemble dictionary ────────────────────────────────────────────
    params = {
        'output_block_dim':             output_dim,
        'output_block_slices':          output_slices,
        'num_nodes_per_hidden_layer':   hidden_width,
        'output_dim':                   n,
        'num_layers':                   num_layers,

        # priors / regularisation constants
        'weight_prior_std_multiplier':  1e-2,
        'likelihood_var_prior_mean':    1e-3,
        'likelihood_var_prior_std':     1e-4,
    }
    return params



def normal(shape, multiplier):
    return dist.Normal(jnp.zeros(shape), multiplier * jnp.ones(shape))

def time_based_decay_schedule(initial_learning_rate, decay_rate):
    def schedule(step):
        return initial_learning_rate / (1 + decay_rate * step)
    return schedule

def get_minibatches_unsupervised(X, Y, batch_size):
    assert X.shape[0] == Y.shape[0]
    N = X.shape[0]
    for i in range(0, N, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]
        
def get_minibatches_supervised(X, Y, Z, batch_size):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[0] == Z.shape[0]
    N = X.shape[0]
    for i in range(0, N, batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size], Z[i:i+batch_size]

def validate_model(key,
                   X_val_norm,X_val,Y_val,
                   problem, params, model_fn, guide_fn):
    key, subkey = jax.random.split(key)
    # helper handles argument order expected by the model
    predictive = Predictive(
        model= model_fn,
        guide=guide_fn,
        params=params,              # overrides sample sites with MAP/VI params
        num_samples=500,
        return_sites=["Y_y"],
    )

    samples = predictive(
        subkey,
        X_val_norm,                     # X_norm
        X_val,                          # X_raw
        Y=None,                         # no ground‑truth
        problem_data=problem,
        vi_parameters=params,      # passed so model has access if needed
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



