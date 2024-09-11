import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn
from typing import Callable


def real_to_complex(x: jax.Array) -> jax.Array:
    """
    Converts a complex array where real and imag parts are stored seperately into a complex numbered array:
    input: x: (...,2)
    output: x: (...) (complex)
    """
    return x[...,0] + 1j * x[...,1]


def complex_to_real(x: jax.Array) -> jax.Array:
    """
    Converts a complex array into a real array with real and imag parts are stored seperately:
    input: x: (...) (complex)
    output: x: (..., 2) 
    """
    return jnp.stack([x.real, x.imag], axis=-1)


def init_VinvCV(rng_key, C_init_fn: Callable, V: jax.Array, Vinv: jax.Array) -> jax.Array:
    """
    Initializes C: Vinv @ weight @ V
    Input:
        rng_key
        C_init_fn: from which the weights are drawn
        V (complex): eigenvectors of Lamba, or Identity (H, H)
        Vinv (complex): inverse eigenvectors of Lambda or Identity (P, P)
    Output:
        C: (real): (P, H, 2)
    """
    in_dim = V.shape[0]
    out_dim = Vinv.shape[1]

    C = C_init_fn(rng_key, (out_dim, in_dim, 2))
    C = real_to_complex(C)
    C = Vinv @ C @ V
    C = complex_to_real(C)
    return C


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