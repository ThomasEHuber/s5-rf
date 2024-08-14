import equinox as eqx
import optax 
from resonator_s5.classifier import Classifier
import jax


def init_optimizer(model: Classifier, standard_lr: float, ssm_lr: float, weight_decay: float, decay_steps: int) -> tuple[optax.GradientTransformation, optax.OptState]: 
    param_spec = jax.tree.map(lambda _: "standard", model) # every jax.Array is associated with label "standard"
        
    def where_params_with_different_lr(pytree: Classifier) -> list[jax.Array]:
        params = []
        for neuron_layer in pytree.neuron_layers:
            params.append(neuron_layer.Lambda)
            # params.append(neuron_layer.ssm.B)
            params.append(neuron_layer.log_step)
            # params.append(neuron_layer.norm)
        return params

    param_spec = eqx.tree_at(
        where_params_with_different_lr,     # returns params that should be associated with different label
        param_spec,                         # our model with associated label
        replace_fn=lambda _: "ssm"          # how the labels should change
    )

    standard_scheduler = optax.cosine_decay_schedule(standard_lr, decay_steps=decay_steps, alpha=1e-6)
    ssm_scheduler = optax.cosine_decay_schedule(ssm_lr, decay_steps=decay_steps, alpha=1e-6)

    optim = optax.multi_transform(
        {
            "standard": optax.inject_hyperparams(optax.adamw)(learning_rate=standard_scheduler, weight_decay=weight_decay),    # adamw: applies weight decay
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_scheduler)                # adam: no weight decay
        },
        param_spec
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    return optim, opt_state