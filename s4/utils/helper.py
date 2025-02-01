from flax import linen as nn
import jax.numpy as np
import jax
from functools import partial

def log_step_initializer(dt_min = 0.001, dt_max = 0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

    return init

def clone_layer(layer):
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1}, 
        split_rngs={"params": True}, # Each layer gets their seperate rng keys for initializing parameters
    )

def map_nested_fn(fn):
    """Recursively apply `fn to the key-value pairs of a nested dict / pytree."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn

@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes = logits.shape[0])
    return -np.sum(one_hot_label*logits)

@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label