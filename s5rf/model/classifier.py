import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from model.resonator import RF, RFDense, LI
from util.ssm_init import init_A

class Classifier(eqx.Module):
    dense_layers: list[RFDense]
    neuron_layers: list[RF]
    drop: eqx.nn.Dropout
    output_dense: RFDense
    li: LI
    
    apply_skip: bool
    dense_dropout: bool


    def __init__(
            self, 
            rng_key,
            input_dim: int,
            output_dim: int,
            num_neurons: list[int],
            num_blocks: list[int],
            eta_min: float,
            eta_max: float,
            activation: str,
            discretization: str,
            keep_imag: bool,
            apply_skip: bool,
            dropout: float,
            dense_dropout: bool = True,
        ) -> None:
        super().__init__()
        assert len(num_blocks) == len(num_neurons)
        self.apply_skip = apply_skip
        self.dense_dropout = dense_dropout
    
        self.dense_layers = []
        self.neuron_layers = []
        prev_V = jnp.eye(input_dim)
        for i, (neurons, blocks) in enumerate(zip(num_neurons, num_blocks)):
            rng_key, dense_key, rf_key = jax.random.split(rng_key, num=3)
            Lambda, V, Vinv = init_A(int(neurons/blocks), blocks)
            V = V if discretization == "zoh" and i == 0 else jnp.eye(V.shape[0])
            
            rf_dense = RFDense(
                dense_key,
                in_dim=prev_V.shape[0],
                out_dim=Vinv.shape[1],
                Vinv=Vinv,
                keep_imag=keep_imag,
            )
            rf_neurons = RF(
                rf_key,
                Lambda=Lambda,
                V=V,
                eta_min=eta_min,
                eta_max=eta_max,
                keep_imag=keep_imag,
                discretization=discretization if i == 0 else "dirac",
                activation=activation,
            )
            prev_V = V
            self.dense_layers.append(rf_dense)
            self.neuron_layers.append(rf_neurons)
            self.drop = eqx.nn.Dropout(dropout)

            rng_key, linear_key = jax.random.split(rng_key, num=2)
            self.output_dense = RFDense(
                linear_key,
                in_dim=V.shape[0],
                out_dim=output_dim,
                Vinv=jnp.eye(output_dim),
                keep_imag=False,
            )
            self.li = LI(dim=output_dim)


    def forward(self, x: jax.Array, rng_key) -> tuple[jax.Array, jax.Array]:
        """
        input: (L/T, H)
        output: (H)
        """
        for i, (dense, neuron) in enumerate(zip(self.dense_layers, self.neuron_layers)):
            rng_key, drop_key = jax.random.split(rng_key, num=2)
            if self.apply_skip and i != 0:
                skip = x

            x = jax.vmap(dense)(x)
            if self.dense_dropout and i != 0 and i < len(self.neuron_layers) - 1:
                x = self.drop(x, key=drop_key)

            x = neuron(x)
            if not self.dense_dropout:
                x = self.drop(x, key=drop_key)
                
            if self.apply_skip and i != 0 and i < len(self.neuron_layers) - 1:
                x = x + skip
        
        x = jax.vmap(self.output_dense)(x)
        x = self.li(x)
        x = jnp.mean(x, axis=0) # pooling
        x = jax.nn.log_softmax(x)
        return x
    
    
    def gen_spikes(self, x: jax.Array) -> jax.Array:
        total_sum_spikes = []
        for i, (dense, neuron) in enumerate(zip(self.dense_layers, self.neuron_layers)):
            if self.apply_skip and i != 0:
                skip = x

            x = jax.vmap(dense)(x)

            spikes = neuron(x)
            total_sum_spikes.append(spikes.sum())
                
            if self.apply_skip and i != 0 and i < len(self.neuron_layers) - 1:
                x = spikes + skip
            else:
                x = spikes

        total_sum_spikes = jnp.asarray(total_sum_spikes)
        return total_sum_spikes
    

    def gen_full_spikes(self, x: jax.Array) -> list[jax.Array]:
        total_spikes = []
        for i, (dense, neuron) in enumerate(zip(self.dense_layers, self.neuron_layers)):
            if self.apply_skip and i != 0:
                skip = x

            x = jax.vmap(dense)(x)

            spikes = neuron(x)
            total_spikes.append(spikes.copy())
                
            if self.apply_skip and i != 0 and i < len(self.neuron_layers) - 1:
                x = spikes + skip
            else:
                x = spikes

        return total_spikes