##dnn functions for JAX implementation
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap, random, value_and_grad
from jax.nn import relu
import optax
from acopf import *
from bnncommon import *
from supervisedmodel import *



#A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def nn_output(params, xin):
  # per-example predictions
  activations = xin
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  return jnp.dot(final_w,activations) + final_b

batched_nn_output = vmap(nn_output, in_axes=(None, 0))

@jit
def nn_four_comp_output(params,xin):
  return  jnp.hstack([nn_output(params[k], xin) for k in range(4) ])

batched_four_comp_nn = vmap(nn_four_comp_output, in_axes=(None, 0))

@jit 
def l1_norm_params(params):
    wt =0.0
    for k in range(4):
        for w,b in params[k]:
            wt += jnp.linalg.norm(w, ord=1)
            wt += jnp.linalg.norm(b, ord=1)
    return  wt


def opf_loss_semisupervised(params, X,Y, penalty, relative_penalty = 1.0):
    Y_pred = batched_four_comp_nn(params, X)
    super_loss = jnp.mean(optax.l2_loss(Y_pred, Y)) + penalty["l1"] * l1_norm_params(params)

    cost = jnp.mean(get_objective_value(Y_pred,opf_data)**2)
    eq_violations =  jnp.mean(get_equality_constraint_violations(X,Y_pred, opf_data)**2)
    ineq_violations =  jnp.mean(get_inequality_constraint_violations(Y_pred, opf_data)**2)

    unsup_loss =  penalty["cost"]*cost + penalty["eq"]*eq_violations + penalty["ineq"]*ineq_violations + penalty["l1"]*l1_norm_params(params)
    return sup_loss + relative_penalty*unsup_loss





