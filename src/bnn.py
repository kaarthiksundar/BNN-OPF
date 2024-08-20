from classes import OPFData
import logging 
from typing import Dict
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp
from collections import OrderedDict
from acopf import assess_feasibility

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
              'likelihood_var_prior_std' : 1e-6, 
              'likelihood_var_prior_L_mean' : 1e-4, 
              'likelihood_var_prior_L_std' : 1e-6
              } 
    return params

def normal(shape, multiplier):
    return dist.Normal(jnp.zeros(shape), multiplier * jnp.ones(shape))

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
            w_shape = (input_dim, num_nodes_per_layer)
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
        (name, numpyro.sample(f'l_var_{name}', normal(mean, std))) for name in params['output_block_dim'].keys()
    ])
    slices = params['output_block_slices']
    
    with numpyro.plate('data', size=num_data_points):
        for name in params['output_block_dim'].keys(): 
            numpyro.sample(f'Y_{name}', normal(z[name], likelihood_std[name]).to_event(1), obs=Y[:, slices[name]])
        numpyro.sample('L', normal(L_predict, likelihood_std['pg'] * 0.01), obs=L_true)
     

# initial guide does not require vi_parameters
def guide_supervised_initial(opf_data: OPFData):
    pass