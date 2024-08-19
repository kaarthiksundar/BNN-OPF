import typer
from typing_extensions import Annotated
from pathlib import Path
import logging

from logger import CustomFormatter
from dataloader import load_data
from acopf import *


app = typer.Typer()

@app.command()
def main(
    data_path: Annotated[str, typer.Option('--datapath', '-p')] = './data/', 
    case: Annotated[str, typer.Option('--case', '-c')] = 'pglib_opf_case14_ieee',
    num_train_per_group: Annotated[int, typer.Option('--train', '-r')] = 50, 
    num_test_per_group: Annotated[int, typer.Option('--test', '-e')] = 10,
    num_unsupervised_per_group: Annotated[int, typer.Option('--unsupervised', '-u')] = 100, 
    debug: Annotated[bool, typer.Option()] = False, 
    warn: Annotated[bool, typer.Option()] = False, 
    error: Annotated[bool, typer.Option()] = False, 
    only_dl_flag: Annotated[bool, typer.Option('--onlydl')] = False) -> None:
    
    if (debug and warn) or (warn and error) or (debug and warn): 
        print(f'only one of --debug, --warn, --error flags can be set')
        return 

    log = get_logger(debug, warn, error)
    count = num_train_per_group + num_test_per_group + num_unsupervised_per_group
    if (count > 15000):
        log.error('One group contains only 15000 data points')
        log.error('Ensure (--train) + (--test) + (--unsupervised) <= 15000')
        log.error(f'current value: {count}')
        return
    
    if (Path(data_path + case + '.m').is_file() == False): 
        log.error(f'File {data_path + case}.m does not exist')
        return

    log.info(f'num train per group: {num_train_per_group}')
    opf_data = load_data(
        data_path, case, log, 
        num_train_per_group, 
        num_test_per_group, 
        num_unsupervised_per_group)
    
    log.info('OPFdata class populated and training data set parsed')
    if (only_dl_flag == True):
        log.info(f'Data downloaded and loaded, quitting because of only_dl_flag = {only_dl_flag}')
        return
    
    eq_violation = get_equality_constraint_violations(
        opf_data.X_train, 
        opf_data.Y_train, 
        opf_data, log)
    
    ineq_violation = get_inequality_constraint_violations(
        opf_data.Y_train, opf_data, log
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