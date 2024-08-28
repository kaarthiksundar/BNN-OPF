import typer
from typing_extensions import Annotated
from pathlib import Path
import logging
import json

from logger import CustomFormatter
from dataloader import load_data
from acopf import *
from bnncommon import *
from supervisedmodel import supervised_run
from classes import SampleCounts


app = typer.Typer()

@app.command()
def main(
    data_path: Annotated[str, typer.Option('--datapath', '-p')] = './data/', 
    case: Annotated[str, typer.Option('--case', '-c')] = 'pglib_opf_case57_ieee',
    config_file: Annotated[str, typer.Option('--config', '-o')] = 'config.json',
    num_groups: Annotated[int, typer.Option(
        '--numgroups', '-n', 
        help = 'data is split into 20 groups with each having 15000 data points use in {1, 2, 4, 8, 16}'
        )] = 2, 
    num_train_per_group: Annotated[int, typer.Option(
        '--train', '-r', 
        help = 'num training points per group (use power of 2)'
        )] = 256,#50, 
    num_test_per_group: Annotated[int, typer.Option(
        '--test', '-e', 
        help = 'num testing points per group'
        )] = 50,
    num_unsupervised_per_group: Annotated[int, typer.Option(
        '--unsupervised', '-u', 
        help = 'num unsupervised points per group (use power of 2)')] = 256, #500, 
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
    
    # cli-arg validation
    if (Path(data_path + config_file).is_file() == False): 
        log.error(f'File {data_path + config_file} does not exist')
        return
    
    data = json.load(open(data_path + config_file))
    batch_size = data["batch_size"]
    g = num_groups 
    r = num_train_per_group
    u = num_unsupervised_per_group 
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
    count = r + num_test_per_group + u
    if (count > 15000):
        log.error('One group contains only 15000 data points')
        log.error('Ensure (--train) + (--test) + (--unsupervised) <= 15000')
        log.error(f'current value: {count}')
        return
    if (Path(data_path + case + '.m').is_file() == False): 
        log.error(f'File {data_path + case}.m does not exist')
        return
    log.info(f'case: {case}')
    log.info(f'# training supervised training samples: {int(g*r)}')
    log.info(f'# training unsupervised training samples: {int(g*u)}')
    log.info(f'# testing samples: {int(g * num_test_per_group)}')
    sample_counts = SampleCounts(
        num_groups = g, 
        num_train_per_group = r, 
        num_test_per_group = num_test_per_group, 
        num_unsupervised_per_group = num_unsupervised_per_group, 
        batch_size = batch_size
    )
    
    log.info(f'started parsing OPF data')
    opf_data = load_data(
        data_path, case, log, sample_counts)
    
    log.info('OPFdata class populated and training data set parsed')
    if (only_dl_flag == True):
        log.info(f'Data downloaded and loaded, quitting because of only_dl_flag = {only_dl_flag}')
        return

    log.debug(f'{opf_data.gen_cost}')
    supervised_run(
        opf_data, log, 
        initial_learning_rate = data.get("initial_learning_rate", 1e-3), 
        decay_rate = data.get("decay_rate", 1e-4), 
        max_training_time = data.get("max_training_time", 100.0), 
        max_epochs = data.get("max_epochs", 100)
        )
    

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