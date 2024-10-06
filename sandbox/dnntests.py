import sys
sys.path.append('src')
import typer
from typing_extensions import Annotated
from logger import CustomFormatter
from pathlib import Path
import logging
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.linalg import vector_norm
import matplotlib.pyplot as plt
from dataloader import load_data
from acopf import *
from bnncommon import *
from dnn import *
from supervisedmodel import *
from stopping import *
from sandwiched import run_sandwich
from classes import SampleCounts
from jax import random
from modelio import *


def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


def main(
    data_path: Annotated[str, typer.Option('--datapath', '-p')] = './data/', 
    case: Annotated[str, typer.Option('--case', '-c')] =  'pglib_opf_case118_ieee',
    config_file: Annotated[str, typer.Option('--config', '-o')] = 'config.json',
    num_groups: Annotated[int, typer.Option(
        '--numgroups', '-n', 
        help = 'data is split into 20 groups with each having 15000 data points use in {1, 2, 4, 8, 16}'
        )] = 1, 
    num_train_per_group: Annotated[int, typer.Option(
        '--train', '-r', 
        help = 'num training points per group (provide power of 2)'
        )] = 512, 
    num_test_per_group: Annotated[int, typer.Option(
        '--test', '-e', 
        help = 'num testing points per group'
        )] = 1000,
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
    
    #cli-arg validation
    loaded_cases = ['pglib_opf_case30_ieee', 'pglib_opf_case57_ieee',
                   'pglib_opf_case118_ieee', 'pglib_opf_case500_goc']
    if case not in loaded_cases:
       log.error(f'{case} can be only lie in {loaded_cases}')
       return 
    
    possible_run_types = ['semisupervisedBNN', 'supervisedBNN', 'supervisedDNN']
    if run_type not in possible_run_types: 
       log.error(f'{run_type} can only lie in {possible_run_types}')
       return
    
    if (Path(data_path + config_file).is_file() == False): 
       log.error(f'File {data_path + config_file} does not exist')
       return
    
    data = json.load(open(data_path + config_file))
    batch_size = data["batch_size"]

    
    # follows a 80 % train, 20 % validation and test data is separate
    log.info(f'num groups to dl: {num_groups}')
    split = (0.80, 0.20)
    g = num_groups 
    r = num_train_per_group
    total = math.ceil(r/split[0])
    u = int(r*4.0)
    t = num_test_per_group
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
    X_train = np.asarray(opf_data.X_train)
    Y_train = np.asarray(opf_data.Y_train)
    X_val = torch.tensor(np.asarray(opf_data.X_val))
    Y_val = torch.tensor(np.asarray(opf_data.Y_val))
    #return X_val, Y_val, opf_data
    X_test = torch.tensor(np.asarray(opf_data.X_test))
    Y_test = torch.tensor(np.asarray(opf_data.Y_test))
    dataset = TfData(X_train, Y_train)
    trainloader=DataLoader(dataset=dataset, batch_size=opf_data.batch_size, shuffle=True)
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    hidden_dim = roundup(3*output_dim)
   # num_hidden = 4
    #hidden_dim = roundup(1.2*output_dim)
    num_hidden = 2
    learning_rate = 0.1
    num_epochs = 2000
    model = AdvancedMLP(input_dim, hidden_dim, num_hidden,  output_dim, drop_rate = 0.7)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epochs, power=1)
    lambda_l1 = 0.0001
    lambda_cost = 1E-5
    lambda_eq = 1E9 ##NOTE This can be chosen after one step in primal-dual method
    lambda_ineq = 0.0
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    summary(model,(input_dim,))

    costs = getcosts(opf_data)

    losses = []
    val_losses = []
    val_feasibility = []
    val_objs = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, Y_batch in trainloader:
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch) 
            + (lambda_l1/pytorch_total_params)*l1_regularization 
            + lambda_cost*torch.max(opf_cost(outputs, opf_data, *costs))
            + lambda_eq*vector_norm(equality_violations(X_batch, outputs, opf_data), ord=1)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        losses.append(running_loss)
        model.eval()
        Y_pred_val = model(X_val).detach()
        val_loss = criterion(Y_pred_val, Y_val)
        val_mean_feas = np.mean(assess_feasibility(
            np.asarray(X_val), Y_pred_val.numpy(), opf_data))
        val_obj = get_objective_value(Y_pred_val.numpy(), opf_data)
        val_losses.append(val_loss.item())
        val_feasibility.append(val_mean_feas)
        val_objs.append(max(val_obj))


        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], val_loss: {val_loss.item():.4f}, val max obj: {val_objs[-1]:.4f}')

# Plot the loss curve
    model.eval()
    Y_pred = model(X_test)
    test_loss = criterion(Y_pred, Y_test)
    true_cost = get_objective_value(np.asarray(Y_test), opf_data)
    feasibility_violation = assess_feasibility(np.asarray(X_test), Y_pred.detach().numpy(), opf_data)
    test_obj = get_objective_value(Y_pred.detach().numpy(), opf_data)

    obj_err = max(np.abs(true_cost - test_obj))

    print(f'Test MSE: {test_loss.item():.4f}')
    print(f'Test L_inf Objective Error: {obj_err:.4f}')
    print(f'Test L_inf feasibility violation:{np.max(feasibility_violation):.4f}')
    print(f'Test mean feasibility violation:{np.mean(feasibility_violation):.4f}')
    plt.plot(losses,label = "train MSE")
    plt.plot(val_losses, label = "validattion MSE")
    plt.plot(val_feasibility, label = "val mean feasibility")
    plt.plot(val_objs, label = "val max cost")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss/Last batch')
    plt.show()



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


main()


