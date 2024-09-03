from classes import OPFData
from typing import Dict
import numpyro.distributions as dist
import jax.numpy as jnp
from collections import OrderedDict

def get_model_params(opf_data: OPFData) -> Dict:
    ngens = opf_data.get_num_gens() 
    nbuses = opf_data.get_num_buses()
    nloads = opf_data.get_num_loads()
    output_dim = OrderedDict([('pg', ngens), ('qg', ngens), ('vm', nbuses), ('va', nbuses)])
    output_slices = OrderedDict([
        ('pg', slice(ngens)), 
        ('qg', slice(ngens, 2*ngens)), 
        ('vm', slice(2*ngens, 2*ngens + nbuses)), 
        ('va', slice(2*ngens + nbuses, 2*ngens + 2*nbuses))
    ])
    params = { 'output_block_dim' : output_dim, 
              'output_block_slices': output_slices,
              'num_nodes_per_hidden_layer' : 2 * nloads, 
              'output_dim' : 2 * (ngens + nbuses),
              'num_layers' : 2, 
              'weight_prior_std_multiplier' : 1e-2, 
              'likelihood_var_prior_mean' : 1e-5, 
              'likelihood_var_prior_std' : 1e-6
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