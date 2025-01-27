##dnn functions for JAX implementation
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap, random, value_and_grad
from jax.nn import relu
import optax


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


