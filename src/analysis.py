import logging


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

    mse = mean_squared_error(y_predict_mean, opf_data.Y_test)
    log.info(f'total prediction MSE: {mse}')
    
    pg, qg, vm, va = get_output_variables(y_predict_mean, opf_data) 
    pg_t, qg_t, vm_t, va_t = get_output_variables(opf_data.Y_test, opf_data)
    
    mse_pg = mean_squared_error(pg, pg_t)
    log.info(f'pg prediction MSE: {mse_pg}')
    mse_qg = mean_squared_error(qg, qg_t)
    log.info(f'qg prediction MSE: {mse_qg}')
    mse_vm = mean_squared_error(vm, vm_t)
    log.info(f'vm prediction MSE: {mse_vm}')
    mse_va = mean_squared_error(va, va_t)
    log.info(f'va prediction MSE: {mse_va}')
    
    feasibility = assess_feasibility(opf_data.X_test, y_predict_mean, opf_data)
    mse_feasibility = sum(feasibility)/len(feasibility)
    # Initialize feasibility matrix with the appropriate size
    Eq_feasibility_all = np.zeros((100, 1000,2*opf_data.get_num_buses()))  # 100 weights and 1000 test inputs are the ranges for k and m
    # eq_pp,ineq_pp = assess_feasibility_pp(opf_data.X_test, y_predict_mean, opf_data)
    InEq_feasibility_all = np.zeros((100, 1000,4*opf_data.get_num_gens()+2*opf_data.get_num_buses()))  # 100 weights and 1000 test inputs are the ranges for k and m

    # Loop through each index k and m
    for k in range(100):  # Replace K with the appropriate limit for k
            # Assess feasibility for the given indices k and m
        Eq_feasibility_all[k,:, :],InEq_feasibility_all[k,:, :] = assess_feasibility_pp(opf_data.X_test, A[k, :, :], opf_data)
    
    Eq_feasibility = jnp.abs(Eq_feasibility_all).max(axis=2)
    Ineq_feasibility = jnp.abs(InEq_feasibility_all).max(axis=2)

    log.info(f'Feasibility Shape: {Eq_feasibility.shape,Ineq_feasibility.shape}')
    # log.info(f'y vector: {y_predict_mean.shape}')
    min_values = np.min(Eq_feasibility, axis=0)
    min_indices = np.argmin(Eq_feasibility, axis=0) + 1  # Adding 1 to shift index to range 1-100

    # Plotting index (1 to 100) and value together in a graph
    plt.figure(figsize=(10, 6))
    plt.scatter(min_indices, min_values, c='blue', label='Minimum values by column')
    plt.xlabel('Index (1 to 100)')
    plt.ylabel('Minimum Value')
    plt.title('Minimum Values and Corresponding Indices for Each Column')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    pf_residuals = get_equality_constraint_violations(opf_data.X_test, y_predict_mean, opf_data)
    
    real_pf_res, imag_pf_res = jnp.array_split(pf_residuals, 2)
    log.info(f'real power flow eq. residuals (max): {real_pf_res.max()}')
    log.info(f'real power flow eq. residuals (min): {real_pf_res.min()}')
    log.info(f'reactive power flow eq. residuals (max): {imag_pf_res.max()}')
    log.info(f'reactive power flow eq. residuals (min): {imag_pf_res.min()}')
    pg_bound_violations = get_pg_bound_violations(pg, opf_data)
    max_violation = (pg_bound_violations[0].max(), pg_bound_violations[1].max())
    log.info(f'max pg bound violation (l, u): {max_violation}')
    qg_bound_violations = get_qg_bound_violations(qg, opf_data)
    max_violation = (qg_bound_violations[0].max(), qg_bound_violations[1].max())
    log.info(f'max qg bound violation (l, u): {max_violation}')
    vm_bound_violations = get_vm_bound_violations(vm, opf_data)
    max_violation = (vm_bound_violations[0].max(), vm_bound_violations[1].max())
    log.info(f'max vm bound violation (l, u): {max_violation}')


def main():
    file = './output/pglib_opf_case118_ieee_1_64_semisupervisedBNN_config.pkl'

    rng_key, sample_counts, params = read_from_file(file)

    log = get_logger(True, False, False)

    opf_data = load_data('./data/', 'pglib_opf_case118_ieee', log, sample_counts)

    run_test_pp(opf_data, rng_key, params, log)
    
main()


