from dc3classes import ProblemData
from typing import Dict
import numpyro.distributions as dist
import jax.numpy as jnp
from collections import OrderedDict
from typing import Any, Dict 

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
