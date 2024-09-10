from supervisedmodel import *
from unsupervisedmodel import * 
from stopping import *
from bnncommon import *
import time

def run_sandwich(
    opf_data: OPFData, log, 
    config = {},
    rng_key = random.PRNGKey(0)
):
    
    initial_learning_rate = config.get("initial_learning_rate", 1e-3)
    decay_rate = config.get("decay_rate", 1e-4) 
    sandwich_rounds = config.get("sandwich_rounds", 10) 
    max_training_time_per_round = config.get("max_training_time_per_round", 200.0)
    max_training_time = config.get("max_training_time", 1000.0)
    max_epochs = config.get("max_epochs", 200) 
    early_stopping_trigger_supervised = config.get("early_stopping_trigger_supervised", 25) 
    early_stopping_trigger_unsupervised = config.get("early_stopping_trigger_unsupervised", 30)
    patience_supervised = config.get("patience_supervised", 3)
    patience_unsupervised = config.get("patience_unsupervised", 5)
    
    # create early stopping for both the supervised and unsupervised runs
    supervised_early_stopper = PatienceThresholdStoppingCriteria(
        log, patience = patience_supervised)
    unsupervised_early_stopper = PatienceThresholdStoppingCriteria(
        log, patience = patience_unsupervised)
    
    max_time_supervised = 0.4 * max_training_time_per_round 
    max_time_unsupervised = 0.6 * max_training_time_per_round
    supervised_params = []
    unsupervised_params = []
    vi_parameters = None 
    model_params = get_model_params(opf_data)
    remaining_time = max_training_time
    start_time = time.time() 
    
    for round in range(sandwich_rounds): 
        log.info(f'round number: {round + 1}')
        # run supervised 
        run_supervised(
            opf_data, log, 
            initial_learning_rate = initial_learning_rate, 
            decay_rate = decay_rate, 
            max_training_time = min(remaining_time, max_time_supervised), 
            max_epochs = max_epochs, 
            validate_every = early_stopping_trigger_supervised, 
            vi_parameters = vi_parameters, 
            stop_check = supervised_early_stopper, 
            rng_key = rng_key
        )
        validation_loss = supervised_early_stopper.best_loss
        log.info(f'supervised validation loss: {validation_loss}')
        vi_parameters = supervised_early_stopper.vi_parameters
        supervised_params.append(vi_parameters)
        supervised_early_stopper.reset_wait() 
        
        # check overall time
        elapsed = time.time() - start_time
        remaining_time = max_training_time - elapsed
        if time.time() - start_time > max_training_time:
            log.info(f'Maximum training time exceeded at supervised round {round}')
            break
        
        # run unsupervised
        run_unsupervised(
            opf_data, log, 
            initial_learning_rate = initial_learning_rate,
            decay_rate = decay_rate, 
            max_training_time = min(remaining_time, max_time_unsupervised), 
            max_epochs = max_epochs, 
            validate_every = early_stopping_trigger_unsupervised, 
            vi_parameters = vi_parameters, 
            stop_check = unsupervised_early_stopper, 
            rng_key = rng_key
        )
        validation_loss = unsupervised_early_stopper.best_loss 
        log.info(f'unsupervised validation loss: {validation_loss}')
        vi_parameters = unsupervised_early_stopper.vi_parameters 
        for name in model_params['output_block_dim'].keys(): 
            mean_key = f'l_std_{name}_mean'
            std_key = f'l_std_{name}_std'
            vi_parameters[mean_key] = supervised_params[-1][mean_key]
            vi_parameters[std_key] = supervised_params[-1][std_key]
        unsupervised_params.append(vi_parameters)
        unsupervised_early_stopper.reset_wait()
        
        # check overall time
        elapsed = time.time() - start_time
        remaining_time = max_training_time - elapsed
        if time.time() - start_time > max_training_time:
            log.info(f'Maximum training time exceeded at unsupervised round {round}')
            break
        
    return vi_parameters
            

    