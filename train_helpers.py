from resonator_s5.optax_helper import init_optimizer
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import torch
from torch.utils.data import DataLoader

from functools import partial

import wandb
from tqdm import tqdm


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

def one_hot(x: np.ndarray) -> np.ndarray:
    """
    Input: (B)
    Output: (B)
    """
    x = np.eye(NUM_CLASSES)[x]
    return x 


def prep_data(x: torch.Tensor, y: torch.Tensor, training: bool, apply_cutmix: bool) -> tuple[jax.Array, jax.Array]:
    x = x.numpy()
    y = y.numpy()
    y = one_hot(y)
    if training and apply_cutmix: 
        x, y = cut_mix(x, y)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
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
    metrics = {
        'loss': loss_value,
        'accuracy': jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1)),
    }
    return metrics


@eqx.filter_jit
def get_average_spikes(model: Classifier, x: jax.Array, layer: int) -> tuple[jax.Array, jax.Array]:
    spikes_fn = partial(model.gen_spikes, layer=layer)
    spikes = jax.vmap(spikes_fn)(x)
    return (jnp.sum(spikes) / x.shape[0]).astype(jnp.int32), jnp.mean(spikes)


def train_epoch(
        model: Classifier, 
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader, 
        test_dataloader: DataLoader, 
        rng_key, 
        opt_state, 
        track_layer: int,
        apply_cutmix: bool,
    ):
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
        x, y = prep_data(x, y, training=True, apply_cutmix=apply_cutmix)
        rng_key, train_key = jax.random.split(rng_key, num=2)
        model, opt_state, metric = train_step(model, train_key, opt_state, x, y)
        train_metrics['loss'].append(metric['loss'])
        train_metrics['accuracy'].append(metric['accuracy'])

    print("validating")
    inference_model = eqx.nn.inference_mode(model)
    for x, y in tqdm(val_dataloader): 
        x, y = prep_data(x, y, training=False, apply_cutmix=apply_cutmix)
        metric = test_step(inference_model, rng_key, x, y)
        val_metrics['loss'].append(metric['loss'])
        val_metrics['accuracy'].append(metric['accuracy'])

    num_spikes, avg_spikes = get_average_spikes(model, x, layer=track_layer)
    val_metrics['num_spikes'] = num_spikes
    val_metrics['avg_spikes'] = avg_spikes

    print("testing")
    for x, y in tqdm(test_dataloader): 
        x, y = prep_data(x, y, training=False, apply_cutmix=apply_cutmix)
        metric = test_step(inference_model, rng_key, x, y)
        test_metrics['loss'].append(metric['loss'])
        test_metrics['accuracy'].append(metric['accuracy'])
    
    num_spikes, avg_spikes = get_average_spikes(model, x, layer=track_layer)
    test_metrics['num_spikes'] = num_spikes
    test_metrics['avg_spikes'] = avg_spikes

    train_metrics['loss'] = jnp.array(train_metrics['loss']).mean()
    train_metrics['accuracy'] = jnp.array(train_metrics['accuracy']).mean()
    val_metrics['loss'] = jnp.array(val_metrics['loss']).mean()
    val_metrics['accuracy'] = jnp.array(val_metrics['accuracy']).mean()
    test_metrics['loss'] = jnp.array(test_metrics['loss']).mean()
    test_metrics['accuracy'] = jnp.array(test_metrics['accuracy']).mean()

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
        wandb_project: str,
    ) -> Classifier:
    from IPython.display import clear_output

    wandb.init(project=wandb_project)

    track_layer = 0
    for epoch in range(epochs):
        if (epoch % 5) == 0: 
            clear_output(wait=True)
        print(f"Epoch: {epoch+1}/{epochs}")
        model, opt_state, train_metrics, val_metrics, test_metrics = train_epoch(model, train_dataloader, val_dataloader, test_dataloader, rng_key, optim, opt_state, track_layer, apply_cutmix)

        wandb.log({
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
            "Max C real": model.dense_layers[track_layer].C[...,0].max(),
            "Min C real": model.dense_layers[track_layer].C[...,0].min(),
            "Max C imag": model.dense_layers[track_layer].C[...,1].max(),
            "Min C imag": model.dense_layers[track_layer].C[...,1].min(),
            "Max Delta": jnp.exp(model.neuron_layers[track_layer].log_step.max()),
            "Min Delta": jnp.exp(model.neuron_layers[track_layer].log_step.min()),
            "Val Num Spikes": val_metrics['num_spikes'], 
            "Val Avg Spikes": val_metrics['avg_spikes'],
            "Test Num Spikes": test_metrics['num_spikes'], 
            "Test Avg Spikes": test_metrics['avg_spikes'],
            "Learning Rate Standard": opt_state.inner_states['standard'].inner_state.hyperparams['learning_rate'],
            "Learning Rate SSM": opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
            "Max Tau": jnp.exp(model.li.tau.max()),
            "Min Tau": jnp.exp(model.li.tau.min()),
            }
        )


    wandb.finish()
    return model