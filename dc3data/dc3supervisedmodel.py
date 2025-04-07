from dc3classes import ProblemData
from dc3feasibility import assess_feasibility
from typing import Union
import logging 
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
import jax.numpy as jnp
from collections import OrderedDict
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
from stopping import *

def supervised_model(
    X_norm, X, Y = None, 
    problem_data: Union[None, ProblemData] = None, 
    vi_parameters = None): 
    params = get_model_params(problem_data)
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
    z_e = z_e * problem_data.Y_std + problem_data.Y_mean
    L_predict = assess_feasibility(X, z_e, problem_data)
    L_true = assess_feasibility(X, Y, problem_data)
    
    # define likelihood variances 
    mean = params['likelihood_var_prior_mean']
    std = params['likelihood_var_prior_std']
    likelihood_std = OrderedDict([
        (name, numpyro.sample(f'l_std_{name}', dist.Normal(mean, std))) for name in params['output_block_dim'].keys()
    ])
    slices = params['output_block_slices']
    
    with numpyro.plate('data', size=num_data_points):
        with handlers.scale(scale=1.0):
            for name in params['output_block_dim'].keys(): 
                numpyro.sample(f'Y_{name}', dist.Normal(z[name], likelihood_std[name]).to_event(1), obs=Y[:, slices[name]])
            numpyro.sample('L', dist.Normal(L_predict, likelihood_std['y'] * 0.01), obs=L_true)
  
# supervised testing model definition
def supervised_testing_model(
    X_norm, X, Y = None, 
    problem_data: Union[None, ProblemData] = None, 
    vi_parameters = None):
    params = get_model_params(problem_data)
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
    
    # define likelihood variances 
    mean = params['likelihood_var_prior_mean']
    std = params['likelihood_var_prior_std']
    likelihood_std = OrderedDict([
        (name, numpyro.sample(f'l_std_{name}', dist.Normal(mean, std))) for name in params['output_block_dim'].keys()
    ])
    
    with numpyro.plate('data', size=num_data_points):
        for name in params['output_block_dim'].keys(): 
            numpyro.sample(f'Y_{name}', dist.Normal(z[name], likelihood_std[name]).to_event(1), obs=None)


# initial guide does not require vi_parameters
def supervised_guide(
    X_norm, X, Y = None, 
    problem_data: Union[None, ProblemData] = None, 
    vi_parameters = None):
    vi_params = vi_parameters if vi_parameters is not None else dict()
    params = get_model_params(problem_data)

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
            numpyro.sample(f'{block_name}_b{i+1}', dist.Normal(b_mean, b_std))
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
    
    # parameterization of the likelihood std using normal distributions  
    mean = params['likelihood_var_prior_mean']
    std = params['likelihood_var_prior_std']
    # initialize values for parameterization, create parameterization and create samples
    for name in params['output_block_dim'].keys(): 
        mean_key = f'l_std_{name}_mean'
        std_key = f'l_std_{name}_std'
        likelihood_std_mean = numpyro.param(mean_key, vi_params.get(mean_key, mean), constraint = constraints.positive)
        likelihood_std_std = numpyro.param(std_key, vi_params.get(std_key, std), constraint = constraints.positive)
        numpyro.sample(f'l_std_{name}', dist.Normal(likelihood_std_mean, likelihood_std_std))
 
def run_supervised(
    problem_data: ProblemData, log, 
    initial_learning_rate = 1e-3, 
    decay_rate = 1e-4, 
    max_training_time = 200.0, 
    max_epochs = 200, 
    validate_every = 10, 
    vi_parameters = None, 
    stop_check = None, 
    rng_key = random.PRNGKey(0)):
    
    if (stop_check == None):
        log.error('early stopping object has to be provided; cannot be None')
        exit()

    # initialize the optimizer
    learning_rate_schedule = time_based_decay_schedule(initial_learning_rate, decay_rate)
    optimizer = chain(clip(10.0), adam(learning_rate_schedule))
    elbo = TraceMeanField_ELBO()
    elbo_val = TraceMeanField_ELBO(num_particles = 50)
    
    # initialize the stochastic variational inference 
    svi = SVI(
        supervised_model, 
        supervised_guide, 
        optimizer, 
        loss = elbo)
    
    svi_state = svi.init(
        rng_key, 
        problem_data.X_train_norm, 
        problem_data.X_train, 
        init_params = vi_parameters, 
        Y = problem_data.Y_train,
        problem_data = problem_data,
        vi_parameters = vi_parameters)
    
    log.info('SVI initialization complete for supervised round')
    
    start_time = time.time()
    validation_loss = float('inf')
    losses = [] 
    for epoch in range(max_epochs):
        if time.time() - start_time > max_training_time: 
            stop_check.on_epoch_end(epoch, validation_loss, vi_parameters)
            log.info('Maximum training time for supervised round exceeded')
            break 
        batch_losses = [] 
        
        for X_norm, X, Y in get_minibatches_supervised(
            problem_data.X_train_norm, 
            problem_data.X_train, 
            problem_data.Y_train, 
            problem_data.batch_size):                                
            svi_state, loss = svi.update(
                svi_state, 
                X_norm, X, Y = Y, 
                problem_data = problem_data, 
                vi_parameters = vi_parameters)
            batch_losses.append(loss)
        mean_batch_loss = np.mean(batch_losses)
        log.debug(f'epoch: {epoch}, mean loss: {mean_batch_loss}')
        losses.append(mean_batch_loss)
        vi_parameters = svi.get_params(svi_state)
        if (epoch % validate_every == 0) and (epoch != 0): 
            validation_loss = elbo_val.loss(
                rng_key, 
                vi_parameters, 
                supervised_model, 
                supervised_guide, 
                problem_data.X_val_norm, 
                problem_data.X_val,
                Y = problem_data.Y_val, 
                problem_data = problem_data, 
                vi_parameters = vi_parameters
            )
            # validation_loss = run_validation_supervised(
            #     problem_data, 
            #     rng_key, 
            #     vi_parameters,
            #     log
            # )
            log.debug(f'validation loss {validation_loss}')
            stop_check.on_epoch_end(epoch, validation_loss, vi_parameters)
        if stop_check.stop_training == True: 
            break
        if time.time() - start_time > max_training_time:
            # validation_loss = run_validation_supervised(
            #     problem_data, 
            #     rng_key, 
            #     vi_parameters,
            #     log
            # )
            validation_loss = elbo_val.loss(
                rng_key, 
                vi_parameters, 
                supervised_model, 
                supervised_guide, 
                problem_data.X_val_norm, 
                problem_data.X_val,
                Y = problem_data.Y_val, 
                problem_data = problem_data, 
                vi_parameters = vi_parameters
            )
            stop_check.on_epoch_end(epoch, validation_loss, vi_parameters)
            log.info('Maximum training time for supervised round exceeded')
            break
    return
    
def predict_supervised(
        rng_key,
        X_norm: jnp.ndarray,
        X_raw:  jnp.ndarray,
        problem_data,
        vi_parameters: dict,
        num_samples: int = 500,
):
    """
    Draw posterior samples and return the mean / std of the predicted *y*
    in **de‑normalised** space.

    Parameters
    ----------
    rng_key       : jax.random.PRNGKey
    X_norm, X_raw : (B, m) arrays – inputs, normalised & original
    problem_data  : ProblemData   – holds Y_mean / Y_std, etc.
    vi_parameters : dict          – best variational params
    num_samples   : int           – posterior samples to average over

    Returns
    -------
    y_mean : (B, n) array   – mean prediction (original units)
    y_std  : (B, n) array   – posterior std   (original units)
    """
    # use the dedicated testing model if it exists, otherwise fall back
    try:
        model_fn = supervised_testing_model   # noqa: F821
    except NameError:
        model_fn = supervised_model           # noqa: F821

    predictive = Predictive(
        model=model_fn,
        guide=supervised_guide,               # noqa: F821
        params=vi_parameters,
        num_samples=num_samples,
        return_sites=("Y_y",)
    )

    samples = predictive(
        rng_key,
        X_norm,
        X_raw,
        Y=None,                              # no ground‑truth at test time
        problem_data=problem_data,
        vi_parameters=vi_parameters,
    )
    return samples

 
