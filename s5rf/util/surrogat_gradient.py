
import jax
import jax.numpy as jnp
from typing import Callable


def heaviside(x: jax.Array) -> jax.Array:
    """
    adapted from https://github.com/kmheckel/spyx
    """
    return jnp.array(x > 0).astype(jnp.float32)


def add_surrogat_gradient(fwd: Callable, bwd: Callable) -> Callable:
    """
    adapted from https://github.com/kmheckel/spyx
    """
    @jax.custom_gradient
    def forward_with_surrogat_gradient(x):
        return fwd(x), lambda g: g * bwd(x)

    return forward_with_surrogat_gradient


def spike_surrgat_tanh(k: float=1.) -> Callable:
    """
    adapted from https://github.com/kmheckel/spyx
    """
    def grad_tanh(x: jax.Array) -> jax.Array:
        kx = k * x
        return (4 / (jnp.exp(-kx) + jnp.exp(kx))**2).astype(jnp.float32)

    return add_surrogat_gradient(heaviside, grad_tanh)


def spike_surrogat_multi_gaussian(h: float=0.15, sigma: float=0.5, s: float=6.) -> Callable:
    """
    implements multi gaussians as surrogat gradient:
    https://arxiv.org/pdf/2103.12593
    h = 0.1, sigma=0.3, s=4
    """
    def grad_multi_gaussian(x: jax.Array) -> jax.Array:
        gaussian = lambda x, mu, std: jnp.exp(-0.5*((x-mu)/std)**2) # * 1/((2*jnp.pi)**(0.5) * std)
        return (1+h)*gaussian(x, 0, sigma) - h*gaussian(x, sigma, s*sigma) - h*gaussian(x, -sigma, s*sigma)
    
    return add_surrogat_gradient(heaviside, grad_multi_gaussian)


def cartesian_spike(x: jax.Array) -> jax.Array:
    re = x.real
    act_fn = spike_surrogat_multi_gaussian()
    return act_fn(re-1) 


def polar_spike(x: jax.Array) -> jax.Array:
    r = jnp.abs(x)
    act_fn = spike_surrogat_multi_gaussian()
    return act_fn(r-1) 

