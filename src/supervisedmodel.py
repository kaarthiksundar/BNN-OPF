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
from stopping import *

# supervised model definition 
def supervised_model(
    X_norm, X, Y = None, 
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
        with handlers.scale(scale=1.0):
            for name in params['output_block_dim'].keys(): 
                numpyro.sample(f'Y_{name}', dist.Normal(z[name], likelihood_std[name]).to_event(1), obs=Y[:, slices[name]])
            numpyro.sample('L', dist.Normal(L_predict, likelihood_std['pg'] * 0.01), obs=L_true)
     
# supervised testing model definition
def supervised_testing_model(
    X_norm, X, Y = None, 
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
def run_supervised(
    opf_data: OPFData, log, 
    initial_learning_rate = 1e-3, 
    decay_rate = 1e-4, 
    max_training_time = 1000.0, 
    max_epochs = 100, 
    validate_every = 10, 
    vi_parameters = None, 
    stop_check = None):
    
    # initialize the optimizer
    learning_rate_schedule = time_based_decay_schedule(initial_learning_rate, decay_rate)
    optimizer = chain(clip(10.0), adam(learning_rate_schedule))
    elbo = TraceMeanField_ELBO()
    if stop_check == None: 
        stop_check = PatienceThresholdStoppingCriteria(log)
    
    # initialize the stochastic variational inference 
    svi = SVI(
        supervised_model, 
        supervised_guide, 
        optimizer, 
        loss = elbo)
    
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(
        rng_key, 
        opf_data.X_train_norm, 
        opf_data.X_train, 
        init_params = vi_parameters, 
        Y = opf_data.Y_train,
        opf_data = opf_data,
        vi_parameters = vi_parameters)
    
    log.info('SVI initialization complete for supervised round')
    
    start_time = time.time()
    losses = [] 
    for epoch in range(max_epochs):
        if time.time() - start_time > max_training_time: 
            stop_check.on_epoch_end(epoch, testing_loss, vi_parameters)
            log.info('Maximum training time reached')
            break 
        batch_losses = [] 
        
        for X_norm, X, Y in get_minibatches_supervised(
            opf_data.X_train_norm, 
            opf_data.X_train, 
            opf_data.Y_train, 
            opf_data.batch_size):
            svi_state, loss = svi.update(
                svi_state, 
                X_norm, X, Y = Y, 
                opf_data = opf_data, 
                vi_parameters = vi_parameters)
            batch_losses.append(loss)
        mean_batch_loss = np.mean(batch_losses)
        log.debug(f'epoch: {epoch}, mean loss: {mean_batch_loss}')
        losses.append(mean_batch_loss)
        vi_parameters = svi.get_params(svi_state)
        if epoch % validate_every == 0: 
            testing_loss = run_test(
                opf_data, 
                rng_key, 
                vi_parameters,
                log
            )
            stop_check.on_epoch_end(epoch, testing_loss, vi_parameters)
        if stop_check.stop_training == True: 
            break
        if time.time() - start_time > max_training_time:
            testing_loss = run_test(
                opf_data, 
                rng_key, 
                vi_parameters,
                log
            )
            stop_check.on_epoch_end(epoch, testing_loss, vi_parameters)
            log.info('Maximum training time reached')
            break
    run_validation(opf_data, rng_key, stop_check.vi_parameters, log)
    
    return stop_check
    
    
def run_test(opf_data: OPFData, rng_key, vi_parameters, log):
    predictive = Predictive(
        model = supervised_testing_model, 
        guide = supervised_guide, 
        params = vi_parameters, 
        num_samples = 100, 
        return_sites = ("Y_pg", "Y_qg", "Y_vm", "Y_va"))

    predictions = predictive(
        rng_key, 
        opf_data.X_test_norm, 
        opf_data.X_test,
        Y = opf_data.Y_test,  
        opf_data = opf_data, 
        vi_parameters = vi_parameters)
    
    combined_predictions = jnp.concatenate([
        predictions['Y_pg'],
        predictions['Y_qg'],
        predictions['Y_vm'],
        predictions['Y_va']
        ], axis=-1)
    A = combined_predictions * opf_data.Y_std + opf_data.Y_mean
    
    y_predict_mean = A.mean(0) 
    y_predict_std = A.std(0)
    
    pg, qg, vm, va = get_output_variables(y_predict_mean, opf_data) 
    pg_t, qg_t, vm_t, va_t = get_output_variables(opf_data.Y_test, opf_data)
    
    mse_pg = mean_squared_error(pg, pg_t)
    mse_qg = mean_squared_error(qg, qg_t)
    mse_va = mean_squared_error(va, va_t)
    eq = get_equality_constraint_violations(opf_data.X_test, y_predict_mean, opf_data).sum(axis=1)
    return mse_pg + mse_va + mse_qg + (eq**2).max()
    
def run_validation(opf_data: OPFData, rng_key, vi_parameters, log):
    predictive = Predictive(
        model = supervised_testing_model, 
        guide = supervised_guide, 
        params = vi_parameters, 
        num_samples = 100, 
        return_sites = ("Y_pg", "Y_qg", "Y_vm", "Y_va"))

    predictions = predictive(
        rng_key, 
        opf_data.X_val_norm, 
        opf_data.X_val,
        Y = opf_data.Y_val,  
        opf_data = opf_data, 
        vi_parameters = vi_parameters)
    
    combined_predictions = jnp.concatenate([
        predictions['Y_pg'],
        predictions['Y_qg'],
        predictions['Y_vm'],
        predictions['Y_va']
        ], axis=-1)
    A = combined_predictions * opf_data.Y_std + opf_data.Y_mean
    
    y_predict_mean = A.mean(0) 
    y_predict_std = A.std(0)

    mse = mean_squared_error(y_predict_mean, opf_data.Y_test)
    log.debug(f'total prediction MSE: {mse}')
    
    pg, qg, vm, va = get_output_variables(y_predict_mean, opf_data) 
    pg_t, qg_t, vm_t, va_t = get_output_variables(opf_data.Y_test, opf_data)
    
    mse_pg = mean_squared_error(pg, pg_t)
    log.debug(f'pg prediction MSE: {mse_pg}')
    mse_qg = mean_squared_error(qg, qg_t)
    log.debug(f'qg prediction MSE: {mse_qg}')
    mse_vm = mean_squared_error(vm, vm_t)
    log.debug(f'vm prediction MSE: {mse_vm}')
    mse_va = mean_squared_error(va, va_t)
    log.debug(f'va prediction MSE: {mse_va}')
    
    feasibility = assess_feasibility(opf_data.X_test, y_predict_mean, opf_data)
    mse_feasibility = sum(feasibility)/len(feasibility)
    
    pf_residuals = get_equality_constraint_violations(opf_data.X_test, y_predict_mean, opf_data)
    
    real_pf_res, imag_pf_res = jnp.array_split(pf_residuals, 2)
    log.debug(f'real power flow eq. residuals (max): {real_pf_res.max()}')
    log.debug(f'real power flow eq. residuals (min): {real_pf_res.min()}')
    log.debug(f'reactive power flow eq. residuals (max): {imag_pf_res.max()}')
    log.debug(f'reactive power flow eq. residuals (min): {imag_pf_res.min()}')
    pg_bound_violations = get_pg_bound_violations(pg, opf_data)
    max_violation = (pg_bound_violations[0].max(), pg_bound_violations[1].max())
    log.debug(f'max pg bound violation (l, u): {max_violation}')
    qg_bound_violations = get_qg_bound_violations(qg, opf_data)
    max_violation = (qg_bound_violations[0].max(), qg_bound_violations[1].max())
    log.debug(f'max qg bound violation (l, u): {max_violation}')
    vm_bound_violations = get_vm_bound_violations(vm, opf_data)
    max_violation = (vm_bound_violations[0].max(), vm_bound_violations[1].max())
    log.debug(f'max vm bound violation (l, u): {max_violation}')