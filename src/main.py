import typer
from typing_extensions import Annotated
from pathlib import Path
import logging

from logger import CustomFormatter
from dataloader import load_training_data_from_mat


app = typer.Typer()

@app.command()
def main(
    datapath: Annotated[str, typer.Argument()] = './data/case57/', 
    trainingdatafile: Annotated[str, typer.Argument()] = 'FeasiblePairs_Case57.mat', 
    casefile: Annotated[str, typer.Argument()] = 'case57_ieee.m',
    debug: Annotated[bool, typer.Option()] = False, 
    warn: Annotated[bool, typer.Option()] = False, 
    error: Annotated[bool, typer.Option()] = False) -> None:
    
    if (debug and warn) or (warn and error) or (debug and warn): 
        print(f'only one of --debug, --warn, --error flags can be set')
        return 

    log = get_logger(debug, warn, error)
    
    if (Path(datapath + trainingdatafile).is_file() == False): 
        log.error(f'File {datapath + trainingdatafile} does not exist')
        return
    
    if (Path(datapath + casefile).is_file() == False): 
        log.error(f'File {datapath + casefile} does not exist')
        return

    data = load_training_data_from_mat(
        datapath + trainingdatafile, 
        datapath + casefile, 
        log)
    
    

def get_logger(debug, warn, error): 
    log = logging.getLogger('bnn-opf')
    log.setLevel(logging.INFO)
    
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


if __name__ == "__main__":
    app()