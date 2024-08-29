from classes import OPFData
from typing import Union
import logging 
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp
from collections import OrderedDict
from acopf import assess_feasibility
from bnncommon import *
from optax import adam, chain, clip, nadam
from numpyro.infer import Predictive, SVI, Trace_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO
from numpyro import handlers
from jax import random
from jax import jit
import jax
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from acopf import *

def unsupervised_model(
    X_norm, X, 
    opf_data: Union[None, OPFData] = None, 
    vi_parameters = None):
    
    params = get_model_params(opf_data)
    num_data_points, num_inputs = X_norm.shape
    num_layers = params['num_layers']
    num_nodes_per_layer = params['num_nodes_per_hidden_layer']
    
    def create_block(block_name: str): 
        z = X_norm
        std_multiplier = params['weight_prior_std_multiplier']
        input_dim_to_layer = num_inputs
        for i in range(num_layers):
            w_shape = (input_dim_to_layer, num_nodes_per_layer)
            b_shape = num_nodes_per_layer 
            w = numpyro.sample(f'{block_name}_w{i+1}', normal(w_shape, std_multiplier))
            b = numpyro.sample(f'{block_name}_b{i+1}', normal(b_shape, std_multiplier))
            z = jax.nn.relu(jnp.matmul(z, w) + b)
            input_dim_to_layer = num_nodes_per_layer
        w_out_shape = (num_nodes_per_layer, params['output_block_dim'][block_name])
        w_out = numpyro.sample(f'{block_name}_w_out', normal(w_out_shape, std_multiplier))
        z_out = jnp.matmul(z, w_out)
        return z_out
    
    z = OrderedDict([ (name, create_block(name)) for name in params['output_block_dim'].keys() ])
    z_e = jnp.concatenate(list(z.values()), axis=-1)
    z_e = z_e * opf_data.Y_std + opf_data.Y_mean
    L = assess_feasibility(X, z_e, opf_data)
    
    with numpyro.plate('data', size=num_data_points):
        with handlers.scale(scale=1.0/1e13):
            numpyro.sample('L', dist.Normal(L, 1e-14), obs=0.0)
            
# initial guide does not require vi_parameters
def unsupervised_guide(
    X_norm, X, 
    opf_data: Union[None, OPFData] = None, 
    vi_parameters = None):
    
    vi_params = vi_parameters if vi_parameters is not None else dict()
    params = get_model_params(opf_data)
    num_data_points, num_inputs = X_norm.shape
    num_layers = params['num_layers']
    num_nodes_per_layer = params['num_nodes_per_hidden_layer']
    
    def get_name_i(param_name: str, block_name: str, mstd: str, layer_count: int):
        return f'{block_name}_{param_name}{layer_count+1}_{mstd}'
    def get_name(param_name: str, block_name: str, mstd: str): 
        return f'{block_name}_{param_name}_{mstd}'
    
    def create_guide_block(block_name: str):
        z = X_norm
        std_multiplier = params['weight_prior_std_multiplier']
        input_dim_to_layer = num_inputs
        for i in range(num_layers):
            w_shape = (input_dim_to_layer, num_nodes_per_layer)
            b_shape = num_nodes_per_layer 
            # generate names for the parameters
            w_mean_name = get_name_i('w', block_name, 'mean', i)
            w_std_name = get_name_i('w', block_name, 'std', i)
            b_mean_name = get_name_i('b', block_name, 'mean', i) 
            b_std_name = get_name_i('b', block_name, 'std', i)
            # initial values for parameterization
            w_mean_init = vi_params.get(w_mean_name, jnp.zeros(w_shape))
            w_std_init = vi_params.get(w_std_name, std_multiplier * jnp.ones(w_shape))
            b_mean_init = vi_params.get(b_mean_name, jnp.zeros(b_shape))
            b_std_init = vi_params.get(b_std_name, std_multiplier * jnp.ones(b_shape))
            # create parameterization
            w_mean = numpyro.param(w_mean_name, w_mean_init)
            w_std = numpyro.param(w_std_name, w_std_init, constraint = constraints.positive)
            b_mean = numpyro.param(b_mean_name, b_mean_init)
            b_std = numpyro.param(b_std_name, b_std_init, constraint = constraints.positive)
            # create sample
            numpyro.sample(f'{block_name}_w{i+1}', dist.Normal(w_mean, w_std))
            numpyro.deterministic(f'{block_name}_b{i+1}', b_mean)
            # update the input dimension to the next layer
            input_dim_to_layer = num_nodes_per_layer
        # generate name for output layer weight parameters
        w_out_shape = (num_nodes_per_layer, params['output_block_dim'][block_name])
        w_out_mean_name = get_name('w_out', block_name, 'mean')
        w_out_std_name = get_name('w_out', block_name, 'std')
        # initial values for parameterization 
        w_out_mean_init = vi_params.get(w_out_mean_name, jnp.zeros(w_out_shape))
        w_out_std_init = vi_params.get(w_out_std_name, std_multiplier * jnp.ones(w_out_shape))
        # create parameterization 
        w_out_mean = numpyro.param(w_out_mean_name, w_out_mean_init)
        w_out_std = numpyro.param(w_out_std_name, w_out_std_init, constraint = constraints.positive)
        # create sample 
        numpyro.sample(f'{block_name}_w_out', dist.Normal(w_out_mean, w_out_std))
    
    for name in params['output_block_dim'].keys():
        create_guide_block(name)
    
def run_unsupervised(
    opf_data: OPFData, log, 
    initial_learning_rate = 1e-3, 
    decay_rate = 1e-4, 
    max_training_time = 60.0, 
    max_epochs = 100, 
    vi_parameters = None):
    
    # initialize the optimizer
    learning_rate_schedule = time_based_decay_schedule(initial_learning_rate, decay_rate)
    optimizer = chain(clip(10.0), adam(learning_rate_schedule))
    elbo = TraceMeanField_ELBO()
    # initialize the stochastic variational inference 
    svi = SVI(
        supervised_model, 
        supervised_guide, 
        optimizer, 
        loss = elbo)
    
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(
        rng_key, 
        opf_data.X_unsupervised, 
        opf_data.X_unsupervised_norm, 
        init_params = vi_parameters, 
        opf_data = opf_data)
    
    
    log.info('SVI initialization complete')