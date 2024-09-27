import logging

import os

from logger import CustomFormatter
from dataloader import load_data
from acopf import *
from bnncommon import *
from supervisedmodel import *
from stopping import *
from sandwiched import run_sandwich
from classes import SampleCounts
from jax import random
from modelio import *
import matplotlib.pyplot as plt
import glob
import os
import glob
import pickle
import concurrent.futures

def get_logger(debug, warn, error): 
    log = logging.getLogger('bnn-opf')
    log.setLevel(logging.DEBUG)
    
    if (debug == True):
        log.setLevel(logging.DEBUG)
    if (error == True): 
        log.setLevel(logging.ERROR)
    if (warn == True):
        log.setLevel(logging.WARNING)
    
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter()) 
    log.addHandler(ch)
    
    # create file handler
    fh = logging.FileHandler(f'./logs/analysis.log', mode='w')
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh) 
    return log

def run_test_pp(opf_data: OPFData, rng_key, vi_parameters, log):
    results = {}
    predictive = Predictive(
        model = supervised_testing_model, 
        guide = supervised_guide, 
        params = vi_parameters, 
        num_samples = 500, 
        return_sites = ("Y_pg", "Y_qg", "Y_vm", "Y_va"))

    predictions = predictive(
        rng_key, 
        opf_data.X_test_norm, 
        opf_data.X_test,
        Y = None,  
        opf_data = opf_data, 
        vi_parameters = vi_parameters)
    
    combined_predictions = jnp.concatenate([
        predictions['Y_pg'],
        predictions['Y_qg'],
        predictions['Y_vm'],
        predictions['Y_va']
        ], axis=-1)
    A = combined_predictions * opf_data.Y_std + opf_data.Y_mean
    results['A']  = A
    y_predict_mean = A.mean(0) 
    y_predict_std = A.std(0)
    
    results['y_predict_mean'] = y_predict_mean;
    results['y_predict_std'] = y_predict_std;

    mse = mean_squared_error(y_predict_mean, opf_data.Y_test)
    # log.info(f'total prediction MSE: {mse}')
    results['mse'] = mse

    pg, qg, vm, va = get_output_variables(y_predict_mean, opf_data) 
    pg_t, qg_t, vm_t, va_t = get_output_variables(opf_data.Y_test, opf_data)
    
    mse_pg = mean_squared_error(pg, pg_t)
    # log.info(f'pg prediction MSE: {mse_pg}')
    results['mse_pg'] = mse_pg

    mse_qg = mean_squared_error(qg, qg_t)
    # log.info(f'qg prediction MSE: {mse_qg}')
    results['mse_qg'] = mse_qg

    mse_vm = mean_squared_error(vm, vm_t)
    # log.info(f'vm prediction MSE: {mse_vm}')
    results['mse_vm'] = mse_vm

    mse_va = mean_squared_error(va, va_t)
    # log.info(f'va prediction MSE: {mse_va}')
    results['mse_va'] = mse_va
    
    
    results['cost_test'] = get_objective_value(opf_data.Y_test, opf_data)
    
    feasibility = assess_feasibility(opf_data.X_test, y_predict_mean, opf_data)
    mse_feasibility = sum(feasibility)/len(feasibility)
    results['mse_feasibility'] = mse_feasibility
    
    feasibility = assess_feasibility(opf_data.X_test, y_predict_mean, opf_data)
    mse_feasibility = sum(feasibility)/len(feasibility)
    
    Eq_feasibility_mean,InEq_feasibility_mean = assess_feasibility_pp(opf_data.X_test, y_predict_mean, opf_data)
    results['Eq_feasibility_mean'] = Eq_feasibility_mean
    results['Ineq_feasibility_mean'] = InEq_feasibility_mean
    
    # Initialize feasibility matrix with the appropriate size
    Eq_feasibility_all = np.zeros((100, 1000,2*opf_data.get_num_buses()))  # 100 weights and 1000 test inputs are the ranges for k and m
    # eq_pp,ineq_pp = assess_feasibility_pp(opf_data.X_test, y_predict_mean, opf_data)
    InEq_feasibility_all = np.zeros((100, 1000,4*opf_data.get_num_gens()+2*opf_data.get_num_buses()))  # 100 weights and 1000 test inputs are the ranges for k and m
    cost_mean = get_objective_value(y_predict_mean, opf_data)
    # log.info(f'Cost Size: {cost_mean.shape}')
    cost_all = np.zeros((1000,100))
    # Loop through each index k and m
    for k in range(100):  # Replace K with the appropriate limit for k
        # Assess feasibility for the given indices k and m
        Eq_feasibility_all[k,:, :],InEq_feasibility_all[k,:, :] = assess_feasibility_pp(opf_data.X_test, A[k, :, :], opf_data)
        cost_all[:,k] = get_objective_value(A[k, :, :], opf_data)
    # Store All feasibility results
    results['Eq_feasibility_all'] = np.abs(Eq_feasibility_all)
    results['Ineq_feasibility_all'] = np.abs(InEq_feasibility_all)
    
    results['cost_mean'] = cost_mean;
    results['cost_all'] = cost_all;
    
    Eq_feasibility = jnp.abs(Eq_feasibility_all).max(axis=2)
    Ineq_feasibility = jnp.abs(InEq_feasibility_all).max(axis=2)

    # Store feasibility results
    results['Eq_feasibility'] = Eq_feasibility
    results['Ineq_feasibility'] = Ineq_feasibility

    # log.info(f'Feasibility Shape: {Eq_feasibility.shape,Ineq_feasibility.shape}')
    # log.info(f'y vector: {y_predict_mean.shape}')
    min_values_all_eq = jnp.min(Eq_feasibility, axis=0)
    min_values_all_ineq = jnp.min(Ineq_feasibility, axis=0)
    results['min_values_all_eq'] = min_values_all_eq
    results['min_values_all_ineq'] = min_values_all_ineq
    
    min_indices_all = jnp.argmin(Eq_feasibility, axis=0) + 1  # Adding 1 to shift index to range 1-100
    
    # log.info(f'Minimum Value Shape: {min_values_all.shape}')

    Eq_feasibility_mean,InEq_feasibility_mean = assess_feasibility_pp(opf_data.X_test, y_predict_mean, opf_data)
    Eq_feasibility_mean_max =  jnp.abs(Eq_feasibility_mean).max(axis=1)# Max Eq error among all nodes-- real and reactive power
    InEq_feasibility_mean_max =  jnp.abs(InEq_feasibility_mean).max(axis=1)# Max Ineq error among all nodes-- real and reactive power

    results['Eq_feasibility_mean_max'] = Eq_feasibility_mean_max
    results['InEq_feasibility_mean_max'] = InEq_feasibility_mean_max
   # Create a two-column plot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # # First plot - histogram of min_values_all_eq with transparency
    # ax1.hist(min_values_all_eq, label='Projected Feasibility', alpha=1.0)  # Set alpha for transparency
    # ax1.hist(Eq_feasibility_mean_max, label='Mean Feasibility', alpha=0.7)  # Set alpha for transparency

    # ax1.set_title('Equality')
    # ax1.grid(True)
    # ax1.legend()

    # # Second plot - histogram of min_values_all_ineq with transparency
    # ax2.hist(min_values_all_ineq, label='Projected Feasibility', alpha=1.0)  # Set alpha for transparency
    # ax2.hist(InEq_feasibility_mean_max, label='Mean Feasibility', alpha=0.7)  # Set alpha for transparency
    # ax2.set_title('Inequality')
    # ax2.grid(True)
    # ax2.legend()
        
    # plt.tight_layout()
    # plt.show()  
        
    pf_residuals = get_equality_constraint_violations(opf_data.X_test, y_predict_mean, opf_data)
    
    real_pf_res, imag_pf_res = jnp.array_split(pf_residuals, 2)
    # log.info(f'real power flow eq. residuals (max): {real_pf_res.max()}')
    # log.info(f'real power flow eq. residuals (min): {real_pf_res.min()}')
    # log.info(f'reactive power flow eq. residuals (max): {imag_pf_res.max()}')
    # log.info(f'reactive power flow eq. residuals (min): {imag_pf_res.min()}')
    results['real_pf_res_max'] = real_pf_res.max()
    results['real_pf_res_min'] = real_pf_res.min()
    results['imag_pf_res_max'] = imag_pf_res.max()
    results['imag_pf_res_min'] = imag_pf_res.min()
    # log.info(f'max vm bound violation (l, u): {max_violation}')
    pg_bound_violations = get_pg_bound_violations(pg, opf_data)
    results['pg_bound_violations_max'] = (pg_bound_violations[0].max(), pg_bound_violations[1].max())

    qg_bound_violations = get_qg_bound_violations(qg, opf_data)
    results['qg_bound_violations_max'] = (qg_bound_violations[0].max(), qg_bound_violations[1].max())

    vm_bound_violations = get_vm_bound_violations(vm, opf_data)
    results['vm_bound_violations_max'] = (vm_bound_violations[0].max(), vm_bound_violations[1].max())

    results['Y_test'] = opf_data.Y_test;

    return results


def compute_and_store_results(R):
    # Compute Mean Percentage Cost Error
    percentage_cost_error_mean = np.mean(np.abs((R['cost_mean']) - (R['cost_test'])) / (R['cost_test'])) *100

    # Compute Equality and Inequality Gaps (Means and Max)
    # Eq_gap_norm_mean = np.linalg.norm(R['Eq_feasibility_mean'], axis=1)
    Eq_gap_mean = np.abs(R['Eq_feasibility_mean']) # Just need to get max and mean without the norm 
    Eq_gap_mean_max = np.mean(Eq_gap_mean.max(axis=1)) # Average violation over instances of the maximum over constraints
    Eq_gap_mean_mean = np.mean(Eq_gap_mean) # Avearge violation over instances and over constraints

    # Ineq_gap_norm_mean = np.linalg.norm(R['Ineq_feasibility_mean'], axis=1) 
    Ineq_gap_norm_mean = (R['Ineq_feasibility_mean']) # Just need to fix max and mean without the norm 
    Ineq_gap_mean_max =np.mean(Ineq_gap_norm_mean.max(axis=1)) # Average violation over instances of the maximum over constraints
    Ineq_gap_mean_mean = np.mean(Ineq_gap_norm_mean)  # Avearge violation over instances and over constraints

    # Compute Gaps for All Entries
    Eq_gap_norm_all = np.linalg.norm(R['Eq_feasibility_all'], axis=2).T
    Ineq_gap_norm_all = np.linalg.norm(R['Ineq_feasibility_all'], axis=2).T
    
    # Lagrangian
    Lagrangian_all = R['cost_all'] + 10000 * Eq_gap_norm_all + 10000 * Ineq_gap_norm_all
    
    # Indices for the Best
    Best_Cost_index = jnp.argmin(R['cost_all'], axis=1) # Lowest value of objective
    Best_Lagrangian_index = jnp.argmin(Lagrangian_all, axis=1)
    
    Eq_gap_max_all = R['Eq_feasibility_all'].max(axis=2).T # Max over constraints Dimension after transpose= Test x Weights
    Ineq_gap_max_all = R['Ineq_feasibility_all'].max(axis=2).T # Max over constraints 

    Best_Eq_gap_index = jnp.argmin(Eq_gap_max_all, axis=1) # Argmin over the weights
    Best_Ineq_gap_index = jnp.argmin(Ineq_gap_max_all, axis=1) # Argmin over the weights
    
    # Number of rows
    n_rows = R['cost_all'].shape[0]

    # Create an array of row indices
    row_indices = np.arange(n_rows)

    # Compute Costs for the Best Indices using advanced indexing
    Cost_Best_Cost = R['cost_all'][row_indices, Best_Cost_index]
    Cost_Best_Lagrangian = R['cost_all'][row_indices, Best_Lagrangian_index]
    Cost_Best_Eq_gap = R['cost_all'][row_indices, Best_Eq_gap_index]
    Cost_Best_Ineq_gap = R['cost_all'][row_indices, Best_Ineq_gap_index]

    # Compute Gaps for the Best Indices (Eq_gap)
    Eq_gap_Best_Cost =  R['Eq_feasibility_all'][Best_Cost_index, np.arange(R['cost_all'].shape[0]), :]
    Eq_gap_Best_Lagrangian =  R['Eq_feasibility_all'][Best_Lagrangian_index, np.arange(R['cost_all'].shape[0]), :]
    Eq_gap_Best_Eq_gap = R['Eq_feasibility_all'][Best_Eq_gap_index, np.arange(R['cost_all'].shape[0]), :]
    
    Ineq_gap_Best_Eq_gap = R['Ineq_feasibility_all'][Best_Eq_gap_index, np.arange(R['cost_all'].shape[0]), :]
    Ineq_gap_Best_Lagrangian = R['Ineq_feasibility_all'][Best_Lagrangian_index, np.arange(R['cost_all'].shape[0]), :]



    # Mmaximum over constraints, minimum over weights and Average violation over instances of the
    Eq_gap_Best_Cost_max = np.mean(Eq_gap_Best_Cost.max(axis=1)) 
    Eq_gap_Best_Lagrangian_max = np.mean(Eq_gap_Best_Lagrangian.max(axis=1))
    Eq_gap_Best_Eq_gap_max = np.mean(Eq_gap_Best_Eq_gap.max(axis=1))
    Eq_gap_Best_Eq_gap_mean = np.mean(Eq_gap_Best_Eq_gap.mean(axis=1))
    Ineq_gap_Best_Eq_gap_max = np.mean(Ineq_gap_Best_Eq_gap.max(axis=1))
    Ineq_gap_Best_Eq_gap_mean = np.mean(Ineq_gap_Best_Eq_gap.mean(axis=1))

    Ineq_gap_Best_Lagrangian_max = np.mean(Ineq_gap_Best_Lagrangian.max(axis=1))
    Ineq_gap_Best_Lagrangian_mean = np.mean(Ineq_gap_Best_Lagrangian.mean(axis=1))
    
    # Percentage Cost Error for Best Indices
    percentage_cost_error_Best_Cost = np.mean(np.abs(Cost_Best_Cost - (R['cost_test'])) / (R['cost_test'])) *100
    percentage_cost_error_Best_Lagrangian = np.mean(np.abs(Cost_Best_Lagrangian - (R['cost_test'])) / (R['cost_test'])) *100
    percentage_cost_error_Best_Eq_gap = np.mean(np.abs(Cost_Best_Eq_gap - (R['cost_test'])) / (R['cost_test'])) *100
    percentage_cost_error_Best_Ineq_gap = np.mean(np.abs(Cost_Best_Ineq_gap - (R['cost_test'])) / (R['cost_test'])) *100
    
    # Store Results in Dictionary (including all gaps and indices)
    results = {
        'Percentage Cost Error (Mean)': percentage_cost_error_mean,
        'Eq Gap Max (Mean)': Eq_gap_mean_max,
        'Eq Gap Mean (Mean)': Eq_gap_mean_mean,
        'Ineq Gap Max (Mean)': Ineq_gap_mean_max,
        'Ineq Gap Mean (Mean)': Ineq_gap_mean_mean,
        'Percentage Cost Error (Best Cost)': percentage_cost_error_Best_Cost,
        'Percentage Cost Error (Best Lagrangian)': percentage_cost_error_Best_Lagrangian,
        'Percentage Cost Error (Best Eq Gap)': percentage_cost_error_Best_Eq_gap,
        'Percentage Cost Error (Best Ineq Gap)': percentage_cost_error_Best_Ineq_gap,
        'Best Cost Index': Best_Cost_index,
        'Best Lagrangian Index': Best_Lagrangian_index,
        'Best Eq Gap Norm Index': Best_Eq_gap_index,
        'Best Ineq Gap Norm Index': Best_Ineq_gap_index,
        'Cost (Best Cost)': Cost_Best_Cost,
        'Cost (Best Lagrangian)': Cost_Best_Lagrangian,
        'Cost (Best Eq Gap)': Cost_Best_Eq_gap,
        'Eq Gap (Best Cost)': Eq_gap_Best_Cost,
        'Eq Gap (Best Lagrangian)': Eq_gap_Best_Lagrangian,
        'Eq Gap (Best Eq Gap)': Eq_gap_Best_Eq_gap,
        'Ineq Gap Mean (Best Lagrangian)':Ineq_gap_Best_Lagrangian_mean,
        'Ineq Gap Max (Best Lagrangian) ': Ineq_gap_Best_Lagrangian_max,
        # 'Eq Gap Mean (Best Lagrangian)': Eq_gap_Best_Lagrangian_mean,
        'Eq Gap Max (Best Lagrangian) ': Eq_gap_Best_Lagrangian_max,
        'Eq Gap Mean (Best Eq Gap)': Eq_gap_Best_Eq_gap_mean,
        'Eq Gap Max (Best Eq Gap)': Eq_gap_Best_Eq_gap_max,
        'Ineq Gap Max (Best Eq Gap)': Ineq_gap_Best_Eq_gap_max,
        'Ineq Gap Mean (Best Eq Gap)': Ineq_gap_Best_Eq_gap_mean,
    }


    return results


def compute_expected_error_Emp_Bernstein(R,opf_data, delta=0.05):
    '''This one use empirical variance in the Empirical Bernstein Bounds'''

    ng = opf_data.get_num_gens()
    nbus = opf_data.get_num_buses()

    # R = read_from_results(file_results)
    err = np.abs(R['A']-R['Y_test']) 

    empirical_error = err.mean(0).mean(0)# Mean across weights and then across samples

    var_err_post = np.var(err, axis=0)  # Variance across posterior samples
    exp_var_test = np.mean(err, axis=1)  # Expectation across testing samples E_x

    empirical_Var_x = np.mean(var_err_post, axis=0) + np.var(exp_var_test, axis=0)
    
    delta=0.05
    N_x = 1000


    var_err_pg = empirical_Var_x[:ng]
    var_err_qg = empirical_Var_x[ng:2*ng]
    var_err_vm = empirical_Var_x[-2*nbus:-nbus]
    var_err_va = empirical_Var_x[-nbus:]

    empirical_error_pg = empirical_error[:ng]
    empirical_error_qg = empirical_error[ng:2*ng]
    empirical_error_vm = empirical_error[-2*nbus:-nbus]
    empirical_error_va = empirical_error[-nbus:]

    '''As Per the empirical Bernstein paper Thm 4, expected error <= estimated error + epsilon with probability at least 1-delta'''

    eps_pg = np.sqrt((2 * var_err_pg * np.log(2 / delta)) / N_x) + (7 * np.log(2 / delta)) / (3 * (N_x - 1))
    eps_qg = np.sqrt((2 * var_err_qg * np.log(2 / delta)) / N_x) + (7 * np.log(2 / delta)) / (3 * (N_x - 1))
    eps_vm = np.sqrt((2 * var_err_vm * np.log(2 / delta)) / N_x) + (7 * np.log(2 / delta)) / (3 * (N_x - 1))
    eps_va = np.sqrt((2 * var_err_va * np.log(2 / delta)) / N_x) + (7 * np.log(2 / delta)) / (3 * (N_x - 1))

    exp_err_pg = empirical_error_pg + eps_pg
    exp_err_qg = empirical_error_qg + eps_qg
    exp_err_vm = empirical_error_vm + eps_vm
    exp_err_va = empirical_error_va + eps_va

    return exp_err_pg,exp_err_qg,exp_err_vm,exp_err_va



def compute_expected_error_Bernstein_MPV(R,opf_data, delta=0.05):
    '''This one use Mean Predicitve Variance in the Theoretical Bernstein Bounds'''
    ng = opf_data.get_num_gens()
    nbus = opf_data.get_num_buses()

    # R = read_from_results(file_results)
    err = np.abs(R['A']-R['Y_test']) 

    empirical_error = err.mean(0).mean(0)


    mean_Pred_Var = np.mean(np.var(R['A'], axis=0), axis=0) # Mean (across testing points) of the variance of the predictions (across weights)

    delta=0.05
    N_x = 1000


    mean_Pred_Var_pg = mean_Pred_Var[:ng]
    mean_Pred_Var_qg = mean_Pred_Var[ng:2*ng]
    mean_Pred_Var_vm = mean_Pred_Var[-2*nbus:-nbus]
    mean_Pred_Var_va = mean_Pred_Var[-nbus:]

    empirical_error_pg = empirical_error[:ng]
    empirical_error_qg = empirical_error[ng:2*ng]
    empirical_error_vm = empirical_error[-2*nbus:-nbus]
    empirical_error_va = empirical_error[-nbus:]

    '''As Per the empirical Bernstein paper Thm 4, expected error <= estimated error + epsilon with probability at least 1-delta'''
    eps_pg = np.sqrt((2 * mean_Pred_Var_pg*2 * np.log(1 / delta)) / N_x) + (2*2 * np.log(1 / delta)) / (3 * (N_x))
    eps_qg = np.sqrt((2 * mean_Pred_Var_qg*2 * np.log(1 / delta)) / N_x) + (2*2 * np.log(1 / delta)) / (3 * (N_x))
    eps_vm = np.sqrt((2 * mean_Pred_Var_vm*2 * np.log(1 / delta)) / N_x) + (2 * np.log(1 / delta)) / (3 * (N_x))
    eps_va = np.sqrt((2 * mean_Pred_Var_va*2 * np.log(1 / delta)) / N_x) + (2 * np.log(1 / delta)) / (3 * (N_x))


    exp_err_pg = empirical_error_pg + eps_pg
    exp_err_qg = empirical_error_qg + eps_qg
    exp_err_vm = empirical_error_vm + eps_vm
    exp_err_va = empirical_error_va + eps_va

    return exp_err_pg,exp_err_qg,exp_err_vm,exp_err_va



def compute_expected_error_Hoeffdings(R,opf_data, delta=0.05):
    '''This one is the Theoretical Hoeffdings Bounds'''
    ng = opf_data.get_num_gens()
    nbus = opf_data.get_num_buses()

    # R = read_from_results(file_results)
    err = np.abs(R['A']-R['Y_test']) 

    empirical_error = err.mean(0).mean(0) # Mean (across testing points) of the variance of the predictions (across weights)

    delta=0.05
    N_x = 1000


    empirical_error_pg = empirical_error[:ng]
    empirical_error_qg = empirical_error[ng:2*ng]
    empirical_error_vm = empirical_error[-2*nbus:-nbus]
    empirical_error_va = empirical_error[-nbus:]

    eps = np.sqrt((np.log(1 / delta)) / N_x) 

    exp_err_pg = empirical_error_pg + eps
    exp_err_qg = empirical_error_qg + eps
    exp_err_vm = empirical_error_vm + eps
    exp_err_va = empirical_error_va + eps

    return exp_err_pg,exp_err_qg,exp_err_vm,exp_err_va

def empirical_error(R,opf_data):
    # R = read_from_results(file_results)
    err = np.abs(R['y_predict_mean']-R['Y_test']) # Error from Mean prediction 
    ng = opf_data.get_num_gens()
    nbus = opf_data.get_num_buses()
    # empirical_error = err.mean(0).mean(0)
    empirical_error = err.mean(0)
    empirical_error_pg = empirical_error[:ng]
    empirical_error_qg = empirical_error[ng:2*ng]
    empirical_error_vm = empirical_error[-2*nbus:-nbus]
    empirical_error_va = empirical_error[-nbus:]
    
    return empirical_error_pg,empirical_error_qg,empirical_error_vm,empirical_error_va


def error_bounds(R,opf_data,delta=0.05):

    exp_err_pg_list, exp_err_qg_list, exp_err_vm_list, exp_err_va_list = compute_expected_error_Emp_Bernstein(R,opf_data, delta=0.05)
    hexp_err_pg_list, hexp_err_qg_list, hexp_err_vm_list, hexp_err_va_list = compute_expected_error_Hoeffdings(R,opf_data, delta=0.05)
    Bexp_err_pg_list, Bexp_err_qg_list, Bexp_err_vm_list, Bexp_err_va_list = compute_expected_error_Bernstein_MPV(R,opf_data, delta=0.05)
    empirical_error_pg,empirical_error_qg,empirical_error_vm,empirical_error_va = empirical_error(R,opf_data)

     # Collect all outputs in a dictionary
    bounds = {
    'Empirical Bernstein': {
        'PG': exp_err_pg_list,
        'QG': exp_err_qg_list,
        'VM': exp_err_vm_list,
        'VA': exp_err_va_list
    },
    'Hoeffding': {
        'PG': hexp_err_pg_list,
        'QG': hexp_err_qg_list,
        'VM': hexp_err_vm_list,
        'VA': hexp_err_va_list
    },
    'Bernstein MPV': {
        'PG': Bexp_err_pg_list,
        'QG': Bexp_err_qg_list,
        'VM': Bexp_err_vm_list,
        'VA': Bexp_err_va_list
    },
    'Empirical Error': {
        'PG': empirical_error_pg,
        'QG': empirical_error_qg,
        'VM': empirical_error_vm,
        'VA': empirical_error_va
    }
    }
    
    return bounds

def compute_error_variances(R):
    """
    Function to compute error variance and mean prediction variance.
    
    Args:
    R (dict): A dictionary containing the posterior samples 'A' and true values 'Y_test'.
    
    Returns:
    dict: A dictionary containing the error variance and mean prediction variance.
    """
    
    # Compute absolute error between predictions (R['A']) and true values (R['Y_test'])
    err = (R['A'] - R['Y_test'])

    # Variance across posterior samples
    var_err_post = np.var(err, axis=0)
    
    # Expectation across testing samples E_x
    exp_var_test = np.mean(err, axis=1)

    # Compute the overall variance error
    var_err = np.mean(var_err_post, axis=0) + np.var(exp_var_test, axis=0)

    # Compute the mean prediction variance
    mean_Pred_Var = np.mean(np.var(R['A'], axis=0), axis=0)
    variance_values = {
        'total_variance_in_error': var_err,
        'mean_predicted_variance': mean_Pred_Var
    }
    # Return the results in a dictionary
    return variance_values
 


def main():
    
        # Get a list of all .pkl files in the output folder
    files = glob.glob('./output/2000-Bus/*.pkl')

    # Iterate over each file
    for file in files:
        print(f'Processing file: {file}')
    
        # Extract the base filename
        base_filename = os.path.basename(file)

        # Split the filename using '_' and re-join the first 4 components
        extracted_part = '_'.join(base_filename.split('_')[:4])

        rng_key, sample_counts, params = read_from_file(file)

        log = get_logger(True, False, False)

        opf_data = load_data('./data/', extracted_part, log, sample_counts)

        res = run_test_pp(opf_data, rng_key, params, log)
        # log.info(f'Result Keys: {res.keys()}')
        res_more = compute_and_store_results(res)
        # log.info(f'Result Keys: {res_more.keys()}')
            # Merge the two dictionaries
        combined_results = {**res, **res_more}
        
        bounds = error_bounds(combined_results,opf_data,delta=0.05)
        variances = compute_error_variances(combined_results)
        combined_results = {**combined_results, **bounds,**variances}
        # log.info(f'Result Keys: {combined_results.keys()}')
        # Extract the filename without the directory and extension
        base_filename = os.path.splitext(os.path.basename(file))[0]

        # Construct the output file path, appending "_results" to the filename
        output_file = f'./results/{base_filename}_results.pkl'

        # Save the combined results in a pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(combined_results, f)

        # Optionally, print a confirmation
        print(f'Results saved to {output_file}')


main()






# def process_file(file):
#     """
#     Processes a single file: extracts data, computes results, and saves the output.
#     This function will be run in parallel for each file.
#     """
#     print(f'Processing file: {file}')
    
#     # Extract the base filename
#     base_filename = os.path.basename(file)

#     # Split the filename using '_' and re-join the first 4 components
#     extracted_part = '_'.join(base_filename.split('_')[:4])

#     # Simulated functions, replace with actual implementations
#     rng_key, sample_counts, params = read_from_file(file)
#     log = get_logger(True, False, False)
#     opf_data = load_data('./data/', extracted_part, log, sample_counts)
    
#     # Run test and store results
#     res = run_test_pp(opf_data, rng_key, params, log)
#     res_more = compute_and_store_results(res)
    
#     # Merge results and compute error bounds/variances
#     combined_results = {**res, **res_more}
#     bounds = error_bounds(combined_results, opf_data, delta=0.05)
#     variances = compute_error_variances(combined_results)
#     combined_results = {**combined_results, **bounds, **variances}
    
#     # Construct the output file path
#     output_file = f'./results/{base_filename}_results.pkl'
    
#     # Save the combined results to a pickle file
#     with open(output_file, 'wb') as f:
#         pickle.dump(combined_results, f)
    
#     print(f'Results saved to {output_file}')


# def main():
#     # Get a list of all .pkl files in the output folder
#     files = glob.glob('./output/118-Bus/*.pkl')
#     # process_file(files)
#     # Run each file processing in parallel using ProcessPoolExecutor
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         executor.map(process_file, files)


# if __name__ == "__main__":
#     main()

