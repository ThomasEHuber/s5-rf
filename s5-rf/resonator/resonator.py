from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.nn.initializers import lecun_normal, normal

from ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal, init_A
from surrogat_gradient import cartesian_spike, polar_spike
from resonator_s5.helpers import complex_to_real, real_to_complex, init_VinvCV


# Discretization functions
def discretize_bilinear(Lambda: jax.Array, B_tilde: jax.Array, Delta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
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


def discretization_exact(Lambda: jax.Array, B_tilde: jax.Array, Delta: jax.Array) -> tuple[jax.Array, jax.Array]:
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



class RadialNorm(eqx.Module):
    """
    normalizes the norm of the complex valued input and keeps the angle 
    """
    norm: eqx.nn.LayerNorm
    keep_imag: bool

    def __init__(self, dim: int, keep_imag: bool) -> None:
        self.norm = eqx.nn.LayerNorm(shape=dim)
        self.keep_imag = keep_imag
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Input: (H, 2)
        Output: (H, 2)
        """
        if self.keep_imag:
            x = real_to_complex(x)
            angle = jnp.angle(x)
            length = jnp.abs(x)
            normed_length = self.norm(length)
            x = normed_length * jnp.exp(1j * angle)
            x = complex_to_real(x)
        else:
            x = self.norm(x)
        return x



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
            disc_fn = discretization_exact
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
    C: jax.Array
    keep_imag: bool
    norm: RadialNorm

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
                self.C = init_VinvCV(rng_key, trunc_standard_normal, V, Vinv)
            else:
                C1_key, C2_key = jax.random.split(rng_key, num=2)
                C1 = init_VinvCV(C1_key, trunc_standard_normal, V, Vinv)
                C2 = init_VinvCV(C2_key, trunc_standard_normal, V, Vinv)
                self.C = jnp.concat([C1, C2], axis=1)
            self.norm = RadialNorm(Vinv.shape[0], keep_imag)


    def __call__(self, x: jax.Array) -> jax.Array:
        """
        input: (H)
        output: (P) or (P, 2) if keep_imag
        """
        x = real_to_complex(self.C) @ x
        if self.keep_imag:
            x = complex_to_real(x)
            return x
        else:
            x = x.real
        x = self.norm(x)
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



class RFS5(eqx.Module):
    # parameters for forward and backward pass
    Lambda: jax.Array 
    V: jax.Array = eqx.field(static=True)
    Vinv : jax.Array
    B: jax.Array | None = None
    C: jax.Array 
    log_step: jax.Array

    # hyperparameters for initialization
    include_B: bool
    keep_imag: bool = True
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    activation: str
    V_pos: str
    discretization: str

    """ The S5 SSM
        Args:
            Lambda (float32): diag state matrix (real and imag)  (P,2)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """
    def __init__(
            self,
            rng_key,
            num_blocks: int,
            input_dim: int, # only used if include_B is True, shitty solution but works
            model_dim: int,
            output_dim: int,            
            dt_min: float,
            dt_max: float,
            include_B: bool,
            conj_sym: bool = True,
            bidirectional: bool = False,
            keep_imag: bool = True,
            step_rescale: float = 1.0,
            activation: str = "cartesian_spike", 
            v_pos: str = "before_spike", # decides where to put V  = "before_spike", "after_spike", "none"
            C_init: str = "lecun", # "lecun", "excluding_V" : Vinv @ B @ C, "including_V" : Vinv @ B @ C @s V 
            discretization: str = "exact", # "exact", "zoh", "bilinear"
    ) -> None:
        super().__init__()
        assert (model_dim % num_blocks) == 0, "H must be divisible by num_blocks"

        self.conj_sym = conj_sym
        self.bidirectional = bidirectional
        self.V_pos = v_pos
        self.include_B = include_B
        self.step_rescale = step_rescale
        self.activation = activation
        self.keep_imag = keep_imag
        self.discretization = discretization

        B_key, C_key, dt_key = jax.random.split(rng_key, num=3)

        self.Lambda, self.V, self.Vinv = init_A(int(model_dim/num_blocks), num_blocks)
        # Lambda_re = jnp.log(-Lambda[...,0])
        # Lambda_im = Lambda[...,1]
        # self.Lambda = jnp.concat([Lambda_re[:, None], Lambda_im[:, None]], axis=-1)

        # might need some clearning...
        if not bidirectional:
            if C_init in ["lecun"]:
                self.C = trunc_standard_normal(C_key, (output_dim, model_dim, 2))
            elif C_init in ["excluding_V"]:
                C = trunc_standard_normal(C_key, (output_dim, model_dim, 2))
                C = (C[...,0] + 1j * C[...,1])
                self.C = jnp.concat([C.real, C.imag], axis=-1) # (output_dim, model_dim, 2)
            elif C_init in ["including_V"]:
                C = trunc_standard_normal(C_key, (output_dim, model_dim, 2))
                C = (C[...,0] + 1j * C[...,1]) @ Vw
                self.C = jnp.concat([C.real, C.imag], axis=-1) # (output_dim, model_dim, 2)
        else:
            C1_key, C2_key = jax.random.split(C_key, num=2)
            if C_init in ["lecun"]:
                C1 = trunc_standard_normal(C1_key, (output_dim, model_dim, 2))
                C2 = trunc_standard_normal(C2_key, (model_dim, model_dim, 2))
                self.C = jnp.concat([C1, C2], axis=1) # (output_dim, 2*model_dim, 2)
            elif C_init in ["exluding_V"]:
                C1 = trunc_standard_normal(C1_key, (output_dim, model_dim, 2))
                C2 = trunc_standard_normal(C2_key, (output_dim, model_dim, 2))
                C1 = (C1[...,0] + 1j * C1[...,1])
                C2 = (C2[...,0] + 1j * C2[...,1])
                C = jnp.concat([C1, C2], axis=1)
                self.C = jnp.concat([C.real, C.imag], axis=-1) # (output_dim, 2*model_dim, 2)
            elif C_init in ["including_V"]:
                C1 = trunc_standard_normal(C1_key, (output_dim, model_dim, 2))
                C2 = trunc_standard_normal(C2_key, (output_dim, model_dim, 2))
                C1 = (C1[...,0] + 1j * C1[...,1]) @ V
                C2 = (C2[...,0] + 1j * C2[...,1]) @ V
                C = jnp.concat([C1, C2], axis=1)
                self.C = jnp.concat([C.real, C.imag], axis=-1) # (output_dim, 2*model_dim, 2)

        self.log_step = init_log_steps(dt_key, (model_dim, dt_min, dt_max))


    def __call__(self, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
            u (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        if self.keep_imag:
            # u is complex:
            u = u[...,0] + 1j * u[...,1] 

        Lambda_tilde = self.Lambda[...,0] + 1j * self.Lambda[...,1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        step = jnp.exp(self.log_step[:, 0])

        if self.discretization in ["exact"]:
            disc_fn = discretization_exact
        elif self.discretization in ["zoh"]:
            disc_fn = discretize_zoh
        elif self.discretization in ["bilinear"]:
            disc_fn = discretize_bilinear
        else:
            raise NotImplementedError("discretization only supports: \"exact\", \"zoh\", and \"bilinear\".")
    
        Lambda_bar, B_bar = disc_fn(Lambda_tilde, jnp.eye(Lambda_tilde.shape[0]), step)

        spikes = apply_ssm(Lambda_bar,
                       B_bar,
                       u,
                       self.bidirectional)
        
        if self.V_pos in ["before_spike"]:
            # xs = jax.vmap(lambda x: self.V @ x)(xs)
            ...

        if self.activation in ["cartesian_spike"]:
            xs = cartesian_spike(xs) 
        elif self.activation in ["polar_spike"]:
            xs = polar_spike(xs)
        else:
            raise NotImplementedError("activation only supports: \"cartesian_spike\" and \"polar_spike\".")
        
        xs = spikes
        if self.V_pos in ["after_spike"]:
            xs = jax.vmap(lambda x: self.V @ x)(xs)


        if self.conj_sym:
            ys = jax.vmap(lambda x: 2*(C_tilde @ x))(xs)
        else:
            ys = jax.vmap(lambda x: (C_tilde @ x))(xs)

        if self.keep_imag:
            ys =  jnp.stack([ys.real, ys.imag], axis=-1)
        else:
            ys = ys.real

        return ys, spikes
        

def main():
    rng_key = jax.random.PRNGKey(0)
    num_blocks = 2
    H = 5
    P = 4
    dt_min = 0.001
    dt_max = 0.1
    conj_sym: bool = True
    bidirectional: bool = False
    step_rescale: float = 1.0
    ssm = RFS5(rng_key, num_blocks, H, P, dt_min, dt_max, conj_sym, bidirectional, step_rescale)

    u = jnp.ones((10, H)).astype(jnp.float32)
    y = ssm(u)
    print(y)
    print(y.shape)

if __name__ == "__main__":
    main()