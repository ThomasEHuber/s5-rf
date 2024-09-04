import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from typing import Callable


def init_weights(linear: nn.Linear, init: Callable, rng_key, zero_bias:bool=True) -> nn.Linear:
    """
    General function to initialize a linear layer 
    """
    weight_key, bias_key = jax.random.split(rng_key, num=2)

    weight_shape = linear.weight.shape
    new_weight = init(weight_key, weight_shape, dtype=jnp.float32)
    bias_shape = linear.bias.shape
    if not zero_bias:
        new_bias = init(bias_key, bias_shape, dtype=jnp.float32)
    else:
        new_bias = jnp.ones(bias_shape)

    new_linear = eqx.tree_at(
        where=lambda l: l.weight,
        pytree=linear,
        replace=new_weight
    )
    new_linear = eqx.tree_at(
        where=lambda l: l.bias,
        pytree=linear,
        replace=new_bias
    )
    return new_linear