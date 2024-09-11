import typer
from typing_extensions import Annotated
from pathlib import Path
import logging
import json
import math

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

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    data_path: Annotated[str, typer.Option('--datapath', '-p')] = './data/', 
    case: Annotated[str, typer.Option('--case', '-c')] = 'pglib_opf_case118_ieee',
    config_file: Annotated[str, typer.Option('--config', '-o')] = 'config.json',
    num_groups: Annotated[int, typer.Option(
        '--numgroups', '-n', 
        help = 'data is split into 20 groups with each having 15000 data points use in {1, 2, 4, 8, 16}'
        )] = 2, 
    num_train_per_group: Annotated[int, typer.Option(
        '--train', '-r', 
        help = 'num training points per group (provide power of 2)'
        )] = 512, 
    run_type: Annotated[str, typer.Option('--runtype')] = 'semisupervisedBNN',  
    track_loss: Annotated[bool, typer.Option(
        '--trackloss', help = 'track all losses for plots')] = False,  
    debug: Annotated[bool, typer.Option(help = 'debug flag')] = False, 
    warn: Annotated[bool, typer.Option(help = 'warn flag')] = False, 
    error: Annotated[bool, typer.Option(help = 'error flag')] = False, 
    only_dl_flag: Annotated[bool, typer.Option(
        '--onlydl', help = 'only download data and exit')] = False) -> None:
    
    if (debug and warn) or (warn and error) or (debug and warn): 
        print(f'only one of --debug, --warn, --error flags can be set')
        return 
    
    log = get_logger(debug, warn, error)
    
    # cli-arg validation
    loaded_cases = ['pglib_opf_case30_ieee', 'pglib_opf_case57_ieee',
                    'pglib_opf_case118_ieee', 'pglib_opf_case500_goc']
    if case not in loaded_cases:
        log.error(f'{case} can be only lie in {loaded_cases}')
        return 
    
    possible_run_types = ['semisupervisedBNN', 'supervisedBNN']
    if run_type not in possible_run_types: 
        log.error(f'{run_type} can only lie in {possible_run_types}')
        return
    
    if (Path(data_path + config_file).is_file() == False): 
        log.error(f'File {data_path + config_file} does not exist')
        return
    
    data = json.load(open(data_path + config_file))
    batch_size = data["batch_size"]
    
    # follows a 75 % train, 15 % validation and 10 % testing data split per group
    split = (0.75, 0.15, 0.10)
    g = num_groups 
    r = num_train_per_group
    total = math.ceil(r/split[0])
    u = int(r*4.0)
    t = math.ceil(total*split[2])
    v = math.ceil(total*split[1])
    b = batch_size
    if ((g & (g - 1) == 0) and g != 0)  == False:
        log.error(f'ensure num groups is a power of 2 and <= 20')
        return
    if ((r & (r - 1) == 0) and r != 0 and r != 1) == False:
        log.error(f'ensure the num train per group is a power of 2 (for batching)')
        return
    if ((b & (b - 1) == 0) and b != 0 and b != 1) == False:
        log.error(f'ensure batch size is a power of 2 (for batching)')
        return
    count = r + t + u + v
    if (count > 15000):
        log.error('One group contains only 15000 data points')
        log.error('train, val, test split is (75, 15, 10)')
        log.error('unsupervised count is (#train * 4)')
        log.error('Adjust train count to ensure train + test + val + unsup <= 15000')
        log.error(f'current count value: {count}')
        return
    if (Path(data_path + case + '.m').is_file() == False): 
        log.error(f'File {data_path + case}.m does not exist')
        return
    log.info(f'case: {case}')
    log.info(f'# training supervised training samples: {int(g*r)}')
    log.info(f'# training unsupervised training samples: {int(g*u)}')
    log.info(f'# testing samples: {int(g*t)}')
    log.info(f'# validation samples: {int(g*v)}')
    
    sample_counts = SampleCounts(
        num_groups = g, 
        num_train_per_group = r, 
        num_test_per_group = t, 
        num_unsupervised_per_group = u, 
        num_validation_per_group = v,
        batch_size = batch_size
    )
    
    log.info(f'started parsing OPF data')
    opf_data = load_data(
        data_path, case, log, sample_counts)
    
    log.info('OPFdata class populated and training data set parsed')
    if (only_dl_flag == True):
        log.info(f'Data downloaded and loaded, quitting because of only_dl_flag = {only_dl_flag}')
        return
    
    rng_key = random.PRNGKey(0)
    vi_parameters = None
    
    if run_type == 'semisupervisedBNN':
        vi_parameters = run_sandwich(
            opf_data, log, 
            config = data,
            rng_key = rng_key
        )
    else: 
        # supervised only 
        initial_learning_rate = data.get("initial_learning_rate", 1e-3)
        decay_rate = data.get("decay_rate", 1e-4) 
        max_training_time = data.get("max_training_time", 1000.0)
        max_epochs = data.get("max_epochs", 200) 
        early_stopping_trigger_supervised = data.get("early_stopping_trigger_supervised", 25) 
        patience_supervised = data.get("patience_supervised", 3)
        supervised_early_stopper = PatienceThresholdStoppingCriteria(
            log, patience = patience_supervised)
        vi_parameters = None 
        run_supervised(
            opf_data, log, 
            initial_learning_rate = initial_learning_rate, 
            decay_rate = decay_rate, 
            max_training_time = max_training_time, 
            max_epochs = max_epochs, 
            validate_every = early_stopping_trigger_supervised, 
            vi_parameters = vi_parameters, 
            stop_check = supervised_early_stopper, 
            rng_key = rng_key
        )
        vi_parameters = unsupervised_early_stopper.vi_parameters 

    config = config_file.split('.')[0]
    output_file = f'./output/{case}_{num_groups}_{num_train_per_group}_{run_type}_{config}.pkl'
    write_to_file(output_file, rng_key, sample_counts, vi_parameters)  
    # run_test(opf_data, rng_key, vi_parameters, log)
    

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
    fh = logging.FileHandler(f'./logs/output.log', mode='w')
    fh.setFormatter(CustomFormatter())
    log.addHandler(fh) 
    return log


if __name__ == "__main__":
    app()