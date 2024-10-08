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

