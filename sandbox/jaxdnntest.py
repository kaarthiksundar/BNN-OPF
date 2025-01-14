import sys
sys.path.append('src')
import typer
from typing_extensions import Annotated
from logger import CustomFormatter
from pathlib import Path
import logging
import json
import math
import numpy as np
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



data_path= './data/'
case=  'pglib_opf_case118_ieee'
config_file= 'config.json'
num_groups = 1
num_train_per_group= 512
num_test_per_group  = 1000

data = json.load(open(data_path + config_file))
batch_size = data["batch_size"]

split = (0.80, 0.20)
g = num_groups
r = num_train_per_group
total = math.ceil(r/split[0])
u = int(r*4.0)
t = num_test_per_group
v = math.ceil(total*split[1])
b = batch_size


count = r + t + u + v
sample_counts = SampleCounts(
    num_groups = g, 
    num_train_per_group = r, 
    num_test_per_group = t, 
    num_unsupervised_per_group = u, 
    num_validation_per_group = v,
    batch_size = batch_size)

log = get_logger(False, False, False)
opf_data = load_data(data_path, case, log, sample_counts)

