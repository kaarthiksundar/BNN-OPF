from classes import OPFData
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
from jax import random
from jax import jit
import jax
from sklearn.metrics import mean_squared_error

# supervised model definition 
def supervised_model(opf_data: OPFData, vi_parameters = None): 
    params = get_model_params(opf_data)
    X_norm = opf_data.X_train_norm  
    X = opf_data.X_train
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
    Y = opf_data.Y_train
    L_predict = assess_feasibility(X, z_e, opf_data)
    L_true = assess_feasibility(X, Y, opf_data)
    
    # define likelihood variances 
    mean = params['likelihood_var_prior_mean']
    std = params['likelihood_var_prior_std']
    likelihood_std = OrderedDict([
        (name, numpyro.sample(f'l_std_{name}', dist.Normal(mean, std))) for name in params['output_block_dim'].keys()
    ])
    slices = params['output_block_slices']
    
    with numpyro.plate('data', size=num_data_points):
        for name in params['output_block_dim'].keys(): 
            numpyro.sample(f'Y_{name}', dist.Normal(z[name], likelihood_std[name]).to_event(1), obs=Y[:, slices[name]])
        numpyro.sample('L', dist.Normal(L_predict, likelihood_std['pg'] * 0.01), obs=L_true)
     
# supervised testing model definition
def supervised_testing_model(opf_data: OPFData, vi_parameters = None):
    params = get_model_params(opf_data)
    X_norm = opf_data.X_test_norm  
    X = opf_data.X_test
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
def supervised_guide(opf_data: OPFData, vi_parameters = None):
    vi_params = vi_parameters if vi_parameters is not None else dict()
    params = get_model_params(opf_data)
    X_norm = opf_data.X_train_norm  
    X = opf_data.X_train
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
    
# run one round of supervised training 
def supervised_run(
    opf_data: OPFData, log, 
    initial_learning_rate = 1e-3, 
    decay_rate = 1e-4, 
    max_training_time = 500.0, 
    max_epochs = 200000):
    
    # initialize the optimizer
    learning_rate_schedule = time_based_decay_schedule(initial_learning_rate, decay_rate)
    optimizer = chain(clip(10.0), adam(learning_rate_schedule))
    
    # initialize the stochastic variational inference 
    svi = SVI(
        supervised_model, 
        supervised_guide, 
        optimizer, 
        loss = Trace_ELBO())
    
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(rng_key, opf_data)
    log.debug(svi.get_params(svi_state))
    # svi_state, loss = svi.update(svi_state, opf_data)
    # log.debug(f'after update: {svi.get_params(svi_state)}')
    # log.debug(loss)
    # update_fn = jit(svi.update)
    log.info('setup of SVI complete')
    # svi_state = svi.run(rng_key, 30, opf_data, init_state = svi_state, init_params = svi.get_params(svi_state)).state
    log.debug(f'X_train: {opf_data.X_train}')
    log.debug(f'Y_train: {opf_data.Y_train}')
    log.debug(f'X_train_norm: {opf_data.X_train_norm}')
    log.debug(f'Y_train_norm: {opf_data.Y_train_norm}')
    for i in range(30):
        svi_state, loss = svi.update(svi_state,opf_data)
        log.debug(f'loss: ${loss}')

    predictive = Predictive(model=supervised_testing_model, guide=supervised_guide, 
                            params=svi.get_params(svi_state), 
                            num_samples=100, return_sites=("Y_pg", "Y_qg", "Y_vm", "Y_va"))

    predictions = predictive(rng_key, opf_data)
    combined_predictions = jnp.concatenate([predictions['Y_pg'],predictions['Y_qg'],predictions['Y_vm'],predictions['Y_va']], axis=-1)
    A = combined_predictions * opf_data.Y_std + opf_data.Y_mean
    
    y_predict_vi_original = A.mean(0) #recover_original_values(y_predict_vi, processed_data['mean_y_train'], processed_data['std_y_train'])
    sigma_y_predict_original = A.std(0)#recover_original_values(sigma_y_predict, processed_data['mean_y_train'], processed_data['std_y_train'])

    mse = mean_squared_error(y_predict_vi_original, opf_data.Y_test)
    print(mse)
    
    # # print(svi_result)
    # epoch_losses = [] 
    # for i in range(10):
    #     svi_state, loss = svi.update(svi_state, opf_data)
    #     epoch_losses.append(loss)
    # log.info(f'epoch losses: {epoch_losses}')
    # # log.info('running')
    # log.info(svi_state)