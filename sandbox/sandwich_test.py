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
case=  'pglib_opf_case57_ieee'
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
    
rng_key = random.PRNGKey(0)
X_train = opf_data.X_train
Y_train = opf_data.Y_train
X_val = opf_data.X_val
Y_val = opf_data.Y_val
#return X_val, Y_val, opf_data
X_test = opf_data.X_test
Y_test = opf_data.Y_test
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
train_size = X_train.shape[0]

hidden_dim = roundup(2*output_dim)
num_hidden = 2
layer_sizes =  [input_dim, *[hidden_dim]*num_hidden, output_dim ]
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
penalty = {
    "l1": 1E-4,
    "cost" :0.0,
    "eq" : 1, ##NOTE This can be chosen after one step in primal-dual method,
    "ineq" : 1
}
PATIENCE = 5
COOLDOWN = 0

FACTOR = 0.3
RTOL = 1e-5
ACCUMULATION_SIZE = 1000

losses = []
val_losses = []
val_feasibility = []
val_objs = []

params = init_network_params(layer_sizes, random.key(0))
#@jit 
def l1_norm_params(params):
    wt =0.0
    for w,b in params:
        wt += jnp.linalg.norm(w, ord=1)
        wt += jnp.linalg.norm(b, ord=1)
    return  wt


#@jit
def opf_loss_supervised(params,X,Y, l1_penalty):
    Y_pred = batched_nn_output(params, X)
    return jnp.mean(optax.l2_loss(Y_pred, Y)) + l1_penalty * l1_norm_params(params)

#@jit
def opf_loss_unsupervised(params, X, penalty):
    Y = batched_nn_output(params, X)
    cost = jnp.mean(get_objective_value(Y,opf_data)**2)
    eq_violations =  jnp.mean(get_equality_constraint_violations(X,Y, opf_data)**2)
    ineq_violations =  jnp.mean(get_inequality_constraint_violations(Y, opf_data)**2)
    return penalty["cost"]*cost + penalty["eq"]*eq_violations + penalty["ineq"]*ineq_violations + penalty["l1"]*l1_norm_params(params)

def opf_loss_semisupervised(params, X,Y, penalty, relative_penalty = 1.0):
    Y_pred = batched_nn_output(params, X)
    super_loss = jnp.mean(optax.l2_loss(Y_pred, Y)) + penalty["l1"] * l1_norm_params(params)

    cost = jnp.mean(get_objective_value(Y_pred,opf_data)**2)
    eq_violations =  jnp.mean(get_equality_constraint_violations(X,Y_pred, opf_data)**2)
    ineq_violations =  jnp.mean(get_inequality_constraint_violations(Y_pred, opf_data)**2)

    unsup_loss =  penalty["cost"]*cost + penalty["eq"]*eq_violations + penalty["ineq"]*ineq_violations + penalty["l1"]*l1_norm_params(params)
    return sup_loss + relative_penalty*unsup_loss


#@jit
def train_step_supervised(params, opt_state, opf_data, X,Y, penalty):

    batch_loss, grads = value_and_grad(opf_loss_supervised)(params,X,Y, penalty["l1"])
    updates, opt_state = optimizer.update(grads, opt_state, params, value=batch_loss)
    params = optax.apply_updates(params, updates)
    return batch_loss, params, opt_state

def train_step_unsupervised(params, opt_state, opf_data, X,Y, penalty):

    batch_loss, grads = value_and_grad(opf_loss_unsupervised)(params,X,penalty)
    updates, opt_state = optimizer.update(grads, opt_state, params, value=batch_loss)
    params = optax.apply_updates(params, updates)
    return batch_loss, params, opt_state



batch_size = 128
num_rounds = 7
optimizer = optax.chain(
        optax.adam(LEARNING_RATE),
        optax.contrib.reduce_on_plateau(
        patience=PATIENCE,
        cooldown=COOLDOWN,
        factor=FACTOR,
        rtol=RTOL,
        accumulation_size=ACCUMULATION_SIZE,
        ),
    )

opt_state = optimizer.init(params)



for T  in range(2*num_rounds + 1):

    for epoch in range(NUM_EPOCHS):
        loss = 0.0
        for bind in range(0,train_size, batch_size):
            X_batch = X_train[bind:bind+batch_size, :]
            Y_batch = Y_train[bind:bind+batch_size, :]
            if T%2 == 0:
                batch_loss, params,opt_state = train_step_supervised(params, opt_state, opf_data, X_batch, Y_batch, penalty)
            else:
                batch_loss, params,opt_state = train_step_unsupervised(params, opt_state, opf_data, X_batch, Y_batch, penalty)


        loss += batch_loss
        Y_pred_val = batched_nn_output(params, X_val)
        #Y_pred_val = Y_val
        val_cost = jnp.max(get_objective_value(Y_pred_val, opf_data))
        val_cost_true = get_objective_value(Y_val, opf_data)
        cost_percent =  jnp.max((get_objective_value(Y_pred_val, opf_data) -  val_cost_true)/val_cost_true)*100
        val_eq_cost = jnp.max(get_equality_constraint_violations(X_val,Y_pred_val, opf_data))
        val_ineq_cost = jnp.max(get_inequality_constraint_violations(Y_pred_val, opf_data))


        val_mse  = jnp.max(optax.l2_loss(Y_pred_val, Y_val))
    print(f'ROUND: {T}, val mse: {val_mse.item():1.3e}, val obj percentage: {cost_percent:1.3e},  val eq cost = {val_eq_cost:1.3e}, val ineq cost = {val_ineq_cost:1.3e}')

Y_pred_test = batched_nn_output(params, X_test)
#Y_pred_test = Y_test
test_cost = jnp.max(get_objective_value(Y_pred_test, opf_data))
test_cost_true = get_objective_value(Y_test, opf_data)
cost_percent =  jnp.max((get_objective_value(Y_pred_test, opf_data) -  test_cost_true)/test_cost_true)*100
test_eq_cost = jnp.max(get_equality_constraint_violations(X_test,Y_pred_test, opf_data))
test_ineq_cost = jnp.max(get_inequality_constraint_violations(Y_pred_test, opf_data))
test_mse  = jnp.max(optax.l2_loss(Y_pred_test, Y_test))

print(f' test mse: {test_mse.item():1.3e}, test obj percentage: {cost_percent:1.3e},  test eq cost = {test_eq_cost:1.3e}, test ineq cost = {test_ineq_cost:1.3e}')



