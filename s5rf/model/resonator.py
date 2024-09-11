from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.nn.initializers import lecun_normal, normal

from ..util.ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, init_A
from ..util.surrogat_gradient import cartesian_spike, polar_spike
from ..util.helpers import complex_to_real, real_to_complex, init_VinvCV


# Discretization functions
def discretize_bilinear(Lambda: jax.Array, B_tilde: jax.Array, Delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        From https://github.com/lindermanlab/S5/blob/main/s5/ssm.py
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda: jax.Array, B_tilde: jax.Array, Delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        From https://github.com/lindermanlab/S5/blob/main/s5/ssm.py
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretization_dirac(Lambda: jax.Array, B_tilde: jax.Array, Delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Discretize a diagonalized, continuous-time linear SSM
        using an exact solution based on spike/delta impulse inputs.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = B_tilde + 1j * jnp.zeros_like(B_tilde)
    return Lambda_bar, B_bar



# Parallel scan operations
@jax.vmap
def binary_operator(q_i: jax.Array, q_j: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        From https://github.com/lindermanlab/S5/blob/main/s5/ssm.py
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar: jax.Array, B_bar: jax.Array, u: jax.Array, bidirectional: bool) -> jax.Array:
    """ Compute the LxH output of discretized SSM given an LxH input.
        From https://github.com/lindermanlab/S5/blob/main/s5/ssm.py
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            u (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones((u.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(u)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator,
                                          (Lambda_elements, Bu_elements),
                                          reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=-1)
    return xs



class RF(eqx.Module):
    Lambda: jax.Array
    V: jax.Array = eqx.field(static=True)
    log_step: jax.Array
    
    keep_imag: bool
    discretization: str
    activation: str
    bidirectional: bool
    step_rescale: float

    def __init__(
        self,
        rng_key,
        Lambda: jax.Array,
        V: jax.Array,
        dt_min: float,
        dt_max: float,
        keep_imag: bool,
        discretization: str,
        activation: str,
        bidirectional: bool,
        step_rescale: float = 1.0,
    ) -> None:
        # self.Lambda = Lambda
        Lambda_real =  Lambda[...,0]
        Lambda_imag =  Lambda[...,1]

        Lambda_real = jnp.log(-Lambda_real)
        
        self.Lambda = jnp.stack([Lambda_real, Lambda_imag], axis=-1)

        self.V = V
        self.log_step = init_log_steps(rng_key, (V.shape[0], dt_min, dt_max))
        
        self.keep_imag = keep_imag
        self.discretization = discretization
        self.activation = activation
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale


    def __call__(self, u: jax.Array) -> jax.Array:
        if self.keep_imag:
            u = real_to_complex(u)
        
        Lambda_real = -jnp.exp(self.Lambda[...,0])
        Lambda_imag = self.Lambda[...,1]
        Lambda_tilde = Lambda_real + 1j * Lambda_imag
        step = jnp.exp(self.log_step[:, 0])

        if self.discretization in ["exact"]:
            disc_fn = discretization_dirac
        elif self.discretization in ["zoh"]:
            disc_fn = discretize_zoh
        elif self.discretization in ["bilinear"]:
            disc_fn = discretize_bilinear
        else:
            raise NotImplementedError("discretization only supports: \"exact\", \"zoh\", and \"bilinear\".")

        Lambda_bar, B_bar = disc_fn(Lambda_tilde, jnp.eye(Lambda_tilde.shape[0]), step)
        
        xs = apply_ssm(
            Lambda_bar,
            B_bar,
            u,
            self.bidirectional
        )

        xs = jax.vmap(lambda x: self.V @ x)(xs)

        if self.activation in ["cartesian_spike"]:
            # spikes = jax.nn.gelu(xs.real)*jax.nn.gelu(xs.imag)
            spikes = cartesian_spike(xs) 
        elif self.activation in ["polar_spike"]:
            spikes = polar_spike(xs)
        else:
            raise NotImplementedError("activation only supports: \"cartesian_spike\" and \"polar_spike\".")
        
        xs = xs.real + xs.imag
        return spikes
    

class RFDense(eqx.Module):
    B: jax.Array
    keep_imag: bool

    def __init__(
            self,
            rng_key,
            V: jax.Array,
            Vinv: jax.Array,
            bidirectional: bool,
            keep_imag: bool,
        ) -> None:
            self.keep_imag = keep_imag
            if not bidirectional:
                self.B = init_VinvCV(rng_key, trunc_standard_normal, V, Vinv)
            else:
                B1_key, B2_key = jax.random.split(rng_key, num=2)
                B1 = init_VinvCV(B1_key, trunc_standard_normal, V, Vinv)
                B2 = init_VinvCV(B2_key, trunc_standard_normal, V, Vinv)
                self.B = jnp.concat([B1, B2], axis=1)


    def __call__(self, x: jax.Array) -> jax.Array:
        """
        input: (H)
        output: (P) or (P, 2) if keep_imag
        """
        x = real_to_complex(self.B) @ x
        if self.keep_imag:
            x = complex_to_real(x)
            return x
        else:
            x = x.real
        return x
    

class LI(eqx.Module):
    tau: jax.Array
    dim: int

    def __init__(self, dim: int) -> None:
        self.tau = jnp.log(jnp.array([0.8]*dim))
        self.dim = dim

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Input: (L/T, H)
        Output: (H)
        """
        x = apply_ssm(jnp.exp(self.tau), jnp.eye(self.dim), x, bidirectional=False)
        return x