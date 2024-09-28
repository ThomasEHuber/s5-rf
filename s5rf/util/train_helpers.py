from model.classifier import Classifier
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import torch
from torch.utils.data import DataLoader

from functools import partial
from typing import Any

import wandb
from tqdm import tqdm


def init_optimizer(model: Classifier, standard_lr: float, ssm_lr: float, weight_decay: float, decay_steps: int) -> tuple[optax.GradientTransformation, optax.OptState]: 
    param_spec = jax.tree.map(lambda _: "standard", model) # every jax.Array is associated with label "standard"
        
    def where_params_with_different_lr(pytree: Classifier) -> list[jax.Array]:
        params = []
        for neuron_layer in pytree.neuron_layers:
            params.append(neuron_layer.Lambda)
            params.append(neuron_layer.log_step)
        return params

    param_spec = eqx.tree_at(
        where_params_with_different_lr,     # returns params that should be associated with different label
        param_spec,                         # the model with associated label
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


def loss_fn(model: Classifier, train_key, x: jax.Array, y: jax.Array):
    call = partial(model.forward, rng_key=train_key)
    batched_model = jax.vmap(call, in_axes=(0), out_axes=(0)) # add batch dim to x only
    logits = batched_model(x)
    loss = optax.softmax_cross_entropy(logits=logits, labels=y)
    loss = loss.mean()
    return loss, (logits)


def cut_mix(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs cut mix augmentation: https://arxiv.org/pdf/1905.04899 but for a sequence.
    Code is deliberately in numpy and not jax.numpy cause it is simpler to work on mutable arrays.
    Input:
        x: (B, T, C)
        y: (B, 1)
    Output:
        x: (B, T, C)
        y: (B, 1)
    """
    B = x.shape[0]
    T = x.shape[1]
    success = False
    while not success:
        lmbda = np.random.uniform(low=0, high=1)
        rand_index = np.random.permutation(np.arange(B))
        
        t1 = np.random.randint(low=0, high=T)
        t2 = int(T * (1-lmbda))
        
        t1_cut = np.clip((t1 - t2) // 2, a_min=0, a_max=T)
        t2_cut = np.clip((t1 + t2) // 2, a_min=0, a_max=T)

        num_spikes2 = x[rand_index, t1_cut:t2_cut, :].sum(axis=(1, 2))
        num_spikes1 = x.sum(axis=(1, 2)) - x[:, t1_cut:t2_cut, :].sum(axis=(1, 2))

        success = np.all((num_spikes2 + num_spikes1) != 0)

    x[:, t1_cut:t2_cut, :] = x[rand_index, t1_cut:t2_cut, :]
    
    lmbda = num_spikes1 / (num_spikes1 + num_spikes2)
    lmbda = lmbda[:, None]
    y = lmbda * y + (1-lmbda) * y[rand_index]
    return x, y


def randomly_shift_data(x: jax.Array) -> jax.Array:
    p = np.random.random((1,), )
    if p <= 0.2:
        x = jnp.roll(x, shift=1, axis=-1)
    elif p >= 0.8: 
        x = jnp.roll(x, shift=-1, axis=-1)
    return x


def prep_data(x: torch.Tensor, y: torch.Tensor, training: bool, apply_cutmix: bool, apply_random_shift: bool) -> tuple[jax.Array, jax.Array]:
    x = x.numpy()
    y = y.numpy()
    if training and apply_cutmix: 
        x, y = cut_mix(x, y)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    if training and apply_random_shift:
        x = randomly_shift_data(x)
    return x, y


@eqx.filter_jit
def train_step(model: Classifier, train_key, optim, opt_state, x: jax.Array, y: jax.Array):
    (loss_value, (logits)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, train_key, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    metrics = {
        'loss': loss_value,
        'accuracy': jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1)),
    }
    return model, opt_state, metrics


@eqx.filter_jit
def test_step(model: Classifier, rng_key, x, y):
    loss_value, (logits) = loss_fn(model, rng_key, x, y)
    avg_spikes = get_avg_spikes(model, x)
    metrics = {
        'loss': loss_value,
        'accuracy': jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1)),
        'avg_spikes': avg_spikes,
    }
    return metrics


@eqx.filter_jit
def get_avg_spikes(model: Classifier, x: jax.Array) -> jax.Array:
    total_sum_spikes = jax.vmap(model.gen_spikes)(x) # (B, num layers)
    avg_spikes = total_sum_spikes.mean(axis=0) # second sum to acount for batch size
    return avg_spikes # (num layers)


def train_epoch(
        model: Classifier, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        rng_key, 
        optim,
        opt_state, 
        apply_cutmix: bool,
        apply_random_shift: bool,
    ) -> tuple[Classifier, Any, dict, dict, dict]:
    train_metrics = {
        'loss': [],
        'accuracy': [],
    }
    val_metrics = {
        'loss': [],
        'accuracy': [],
        'num_spikes': [],
        'avg_spikes': [],
    }
    test_metrics = {
        'loss': [],
        'accuracy': [],
        'num_spikes': [],
        'avg_spikes': [],
    }
    print("training")
    for x, y in tqdm(train_dataloader):
        x, y = prep_data(x, y, training=True, apply_cutmix=apply_cutmix, apply_random_shift=apply_random_shift)
        rng_key, train_key = jax.random.split(rng_key, num=2)
        model, opt_state, metric = train_step(model, train_key, optim, opt_state, x, y)
        train_metrics['loss'].append(metric['loss'])
        train_metrics['accuracy'].append(metric['accuracy'])

    print("validating")
    inference_model = eqx.nn.inference_mode(model)
    for x, y in tqdm(val_dataloader): 
        x, y = prep_data(x, y, training=False, apply_cutmix=apply_cutmix, apply_random_shift=apply_random_shift)
        metric = test_step(inference_model, rng_key, x, y)
        val_metrics['loss'].append(metric['loss'])
        val_metrics['accuracy'].append(metric['accuracy'])
        val_metrics['avg_spikes'].append(metric['avg_spikes'])

    print("testing")
    for x, y in tqdm(test_dataloader): 
        x, y = prep_data(x, y, training=False, apply_cutmix=apply_cutmix, apply_random_shift=apply_random_shift)
        metric = test_step(inference_model, rng_key, x, y)
        test_metrics['loss'].append(metric['loss'])
        test_metrics['accuracy'].append(metric['accuracy'])
        test_metrics['avg_spikes'].append(metric['avg_spikes'])


    train_metrics['loss'] = jnp.array(train_metrics['loss']).mean()
    train_metrics['accuracy'] = jnp.array(train_metrics['accuracy']).mean()
    val_metrics['loss'] = jnp.array(val_metrics['loss']).mean()
    val_metrics['accuracy'] = jnp.array(val_metrics['accuracy']).mean()
    val_metrics['avg_spikes'] = jnp.array(val_metrics['avg_spikes']).mean(axis=0) # (num_layers)
    test_metrics['loss'] = jnp.array(test_metrics['loss']).mean()
    test_metrics['accuracy'] = jnp.array(test_metrics['accuracy']).mean()
    test_metrics['avg_spikes'] = jnp.array(test_metrics['avg_spikes']).mean(axis=0) # (num_layers)

    print(f"Train Loss: {train_metrics['loss']}")
    print(f"Train Acc: {train_metrics['accuracy']}")
    print(f"Val Loss: {val_metrics['loss']}")
    print(f"Val Acc: {val_metrics['accuracy']}")
    print(f"Test Loss: {test_metrics['loss']}")
    print(f"Test Acc: {test_metrics['accuracy']}\n")

    return model, opt_state, train_metrics, val_metrics, test_metrics


def train(
        epochs: int,
        model: Classifier, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        rng_key, 
        optim, 
        opt_state,
        apply_cutmix:bool,
        apply_random_shift: bool,
        use_wandb:bool,
        wandb_project: str,
    ) -> Classifier:

    if use_wandb:
        wandb.init(project=wandb_project)

    track_layer = 0
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        model, opt_state, train_metrics, val_metrics, test_metrics = train_epoch(
            model, 
            train_dataloader, 
            val_dataloader, 
            test_dataloader, 
            rng_key, 
            optim, 
            opt_state, 
            apply_cutmix, 
            apply_random_shift
        )
        
        logs = {
            "Train Loss": train_metrics['loss'], 
            "Train Accuracy": train_metrics['accuracy'], 
            "Val Loss": val_metrics['loss'], 
            "Val Accuracy": val_metrics['accuracy'],
            "Test Loss": test_metrics['loss'], 
            "Test Accuracy": test_metrics['accuracy'],
            "Max Lambda real": -jnp.exp(model.neuron_layers[track_layer].Lambda[...,0].min()),
            "Min Lambda real": -jnp.exp(model.neuron_layers[track_layer].Lambda[...,0].max()),
            "Max Lambda imag": model.neuron_layers[track_layer].Lambda[...,1].max(),
            "Min Lambda imag": model.neuron_layers[track_layer].Lambda[...,1].min(),
            "Max B real": model.dense_layers[track_layer].B[...,0].max(),
            "Min B real": model.dense_layers[track_layer].B[...,0].min(),
            "Max B imag": model.dense_layers[track_layer].B[...,1].max(),
            "Min B imag": model.dense_layers[track_layer].B[...,1].min(),
            "Max Delta": jnp.exp(model.neuron_layers[track_layer].log_step.max()),
            "Min Delta": jnp.exp(model.neuron_layers[track_layer].log_step.min()),
            "Learning Rate Standard": opt_state.inner_states['standard'].inner_state.hyperparams['learning_rate'],
            "Learning Rate SSM": opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
            "Max Tau": jnp.exp(model.li.tau.max()),
            "Min Tau": jnp.exp(model.li.tau.min()),
        }
        for i in range(test_metrics['avg_spikes'].shape[0]):
            if 'avg_spikes' in logs.keys():
                # should only not be in keys on SHD
                logs[f'Val Avg Spikes Layer {i}'] = val_metrics['avg_spikes'][i]
            logs[f'Test Avg Spikes Layer {i}'] = test_metrics['avg_spikes'][i]
        
        if use_wandb:
            wandb.log(logs)

    if use_wandb:
        wandb.finish()
    
    return model