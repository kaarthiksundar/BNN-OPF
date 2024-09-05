from supervisedmodel import *
from unsupervisedmodel import * 
from stopping import *
from bnncommon import *

def run_sandwich(
    opf_data: OPFData, log, 
    initial_learning_rate = 1e-3, 
    decay_rate = 1e-4, 
    sandwich_rounds = 3, 
    max_training_time_per_round = 200.0, 
    max_epochs = 200, 
    early_stopping_trigger_supervised = 25, 
    early_stopping_trigger_unsupervised = 30,
    rng_key = random.PRNGKey(0)
):
    
    # create early stopping for both the supervised and unsupervised runs
    # patience for unsupervised (supervised) run is 5 (3)
    supervised_early_stopper = PatienceThresholdStoppingCriteria(
        log, patience = 3)
    unsupervised_early_stopper = PatienceThresholdStoppingCriteria(
        log, patience = 5)
    
    max_time_supervised = 0.4 * max_training_time_per_round 
    max_time_unsupervised = 0.6 * max_training_time_per_round
    supervised_params = []
    unsupervised_params = []
    vi_parameters = None 
    model_params = get_model_params(opf_data)
    
    for round in range(sandwich_rounds): 
        log.info(f'round number: {round + 1}')
        # run supervised 
        run_supervised(
            opf_data, log, 
            initial_learning_rate = initial_learning_rate, 
            decay_rate = decay_rate, 
            max_training_time = max_time_supervised, 
            max_epochs = max_epochs, 
            validate_every = early_stopping_trigger_supervised, 
            vi_parameters = vi_parameters, 
            stop_check = supervised_early_stopper, 
            rng_key = rng_key
        )
        test_loss = supervised_early_stopper.best_loss
        log.info(f'supervised testing loss: {test_loss}')
        vi_parameters = supervised_early_stopper.vi_parameters
        supervised_params.append(vi_parameters)
        supervised_early_stopper.reset_wait() 
        
        continue
        # run unsupervised
        run_unsupervised(
            opf_data, log, 
            initial_learning_rate = initial_learning_rate,
            decay_rate = decay_rate, 
            max_training_time = max_time_unsupervised, 
            max_epochs = max_epochs, 
            validate_every = early_stopping_trigger_unsupervised, 
            vi_parameters = vi_parameters, 
            stop_check = unsupervised_early_stopper, 
            rng_key = rng_key
        )
        test_loss = unsupervised_early_stopper.best_loss 
        log.info(f'unsupervised testing loss: {test_loss}')
        vi_parameters = unsupervised_early_stopper.vi_parameters 
        for name in model_params['output_block_dim'].keys(): 
            mean_key = f'l_std_{name}_mean'
            std_key = f'l_std_{name}_std'
            vi_parameters[mean_key] = supervised_params[-1][mean_key]
            vi_parameters[std_key] = supervised_params[-1][std_key]
        unsupervised_params.append(vi_parameters)
        unsupervised_early_stopper.reset_wait()
        
    return vi_parameters
            

    