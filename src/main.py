import typer
from typing_extensions import Annotated
from pathlib import Path
import logging

from logger import CustomFormatter
from dataloader import load_training_data_from_mat


app = typer.Typer()

@app.command()
def main(
    data_path: Annotated[str, typer.Argument()] = './data/', 
    case: Annotated[str, typer.Argument()] = 'pglib_opf_case14_ieee',
    debug: Annotated[bool, typer.Option()] = False, 
    warn: Annotated[bool, typer.Option()] = False, 
    error: Annotated[bool, typer.Option()] = False) -> None:
    
    if (debug and warn) or (warn and error) or (debug and warn): 
        print(f'only one of --debug, --warn, --error flags can be set')
        return 

    log = get_logger(debug, warn, error)
    
    if (Path(data_path + case + '.m').is_file() == False): 
        log.error(f'File {data_path + case}.m does not exist')
        return

    data = load_training_data_from_mat(data_path, case, log)
    

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
    return log


if __name__ == "__main__":
    app()