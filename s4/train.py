from s4.utils.dataset import generate_mnist_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from rich.logging import RichHandler
import torch
import jax
import optax
from s4.models.s4_model import S4Layer
from flax.training import checkpoints, train_state
from s4.models.batch_stacked_model import BatchStackedModel
from s4.utils.helper import map_nested_fn, cross_entropy_loss, compute_accuracy
from functools import partial
import jax.numpy as np
from flax.core import unfreeze
from tqdm import tqdm
import shutil
import os


logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")

Models = {
    "s4": S4Layer,
    # "ff": FeedForwardModel,
    # "lstm": LSTMRecurrentModel,
    # "ssm": SSMLayer,
}

Dataset = {
    "mnist": generate_mnist_dataset
}

def create_train_state(
    rng, 
    model_cls, 
    trainloader, 
    lr = 1e-3,
    lr_layer = None,
    lr_schedule = False,
    weight_decay = 0.0,
    total_steps = -1):

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    params = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        np.array(next(iter(trainloader))[0].numpy()),
    )

    params = unfreeze(params["params"])

    if lr_schedule:
        schedule_fn = lambda lr: optax.cosine_onecycle_schedule(
            peak_value = lr,
            transition_steps = total_steps,
            pct_start = 0.1
        )
    
    else:
        schedule_fn = lambda lr:lr

    if lr_layer is None:
        lr_layer = {}
    
    optimizers = {
        k: optax.adam(learning_rate = schedule_fn(v*lr))
        for k,v in lr_layer.items()
    }

    optimizers['__default__'] = optax.adamw(
        learning_rate = schedule_fn(lr),
        weight_decay = weight_decay,
    )

    name_map = map_nested_fn(lambda k, _: k if k in lr_layer else '__default__')
    tx = optax.multi_transform(optimizers, name_map)

    extra_keys = set(lr_layer.keys()) - set(jax.tree_leaves(name_map(params)))
    assert (len(extra_keys) == 0), f"Special params {extra_keys} do not correspond to actual params"

    _is_complex = lambda x: x.dtype in [np.complex64, np.complex128]

    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if _is_complex(param) else 1)
        if lr_layer.get(k, lr) > 0.0
        else 0
    )(params)

    log.info(f"[*] Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")
    log.info(f"[*] Total training steps: {total_steps}")
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

@partial(jax.jit, static_argnums = (4,5))
def train_step(state, rng, batch_inputs, batch_labels, model, classification=False):

    def loss_fn(params):
        logits, mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            rngs = {"dropout": rng},
            mutable = ["intermediates"],
        )

        loss = np.mean(cross_entropy_loss(logits, batch_labels))
        acc = np.mean(compute_accuracy(logits, batch_labels))
        
        return loss, (logits, acc)

    if not classification:
        batch_labels = batch_inputs[:, :, 0]

    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (loss, (logits, acc)), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)

    return state, loss, acc

def train_epoch(state, rng, model, trainloader, classification = False):
    
    model = model(training=True)
    batch_losses, batch_accuracies = [], []

    for batch_idx, (inputs, labels) in enumerate(tqdm(trainloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())

        rng, drop_rng = jax.random.split(rng)

        state, loss, acc = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            model,
            classification = classification
        )

        batch_losses.append(loss)
        batch_accuracies.append(acc)

    return (
        state,
        np.mean(np.array(batch_losses)),
        np.mean(np.array(batch_accuracies)),
    )

@partial(jax.jit, static_argnums=(3, 4))
def eval_step(batch_inputs, batch_labels, params, model, classification=False):
    if not classification:
        batch_labels = batch_inputs[:, :, 0]
    
    logits = model.apply({"params": params}, batch_inputs)
    loss = np.mean(cross_entropy_loss(logits, batch_labels))
    acc = np.mean(compute_accuracy(logits, batch_labels))

    return loss, acc

def validate(params, model, testloader, classification=False):
    model = model(training=False)

    losses, accuracies = [], []

    for batch_idx, (inputs, labels) in enumerate(tqdm(testloader)):
        inputs = np.array(inputs.numpy())
        labels = np.array(labels.numpy())

        loss, acc = eval_step(
            inputs, labels, params, model, classification=classification
        )

        losses.append(loss)
        accuracies.append(acc)

    return np.mean(np.array(losses)), np.mean(np.array(accuracies))

def example_train(dataset: str, layer: str, seed: int, model: DictConfig, train: DictConfig):
    
    if not train.checkpoint:
        log.warning("Models checkpoints are set to False!")

    torch.random.manual_seed(seed)
    key = jax.random.PRNGKey(seed)
    key, rng, train_rng = jax.random.split(key, num=3)

    classification = "classification" in dataset

    create_dataset_fn = Dataset[dataset]

    trainloader, testloader, n_classes, l_max, d_input = create_dataset_fn(
        batch_size=train.batch_size
    )

    layer_cls = Models[layer]

    OmegaConf.set_struct(model, False)

    model['layer']['l_max'] = l_max

    lr_layer = getattr(layer_cls, "lr", None)
    
    n_classes = n_classes

    model_cls = partial(
        BatchStackedModel,
        layer_cls = layer_cls,
        d_output = n_classes,
        classification = classification,
        **model
    )

    state = create_train_state(
        rng,
        model_cls,
        trainloader,
        lr=train.lr,
        lr_layer=lr_layer,
        lr_schedule=train.lr_schedule,
        weight_decay=train.weight_decay,
        total_steps=len(trainloader) * train.epochs
    )

    best_loss, best_acc, best_epoch = 10000, 0, 0

    for epoch in range(train.epochs):
        log.info(f"[*] Starting Training Epoch {epoch + 1}...")

        state, train_loss, train_acc = train_epoch(
            state, 
            train_rng,
            model_cls,
            trainloader,
            classification = classification
        )

        print(f"[*] Running Epoch {epoch + 1} Validation...")

        test_loss, test_acc = validate(
            state.params, model_cls, testloader, classification=classification
        )

        log.info(
            f"\n=>> Epoch {epoch + 1} Metrics ==="
            f"\tTrain Loss: {train_loss:.5f} -- Train Accuracy:"
            f" {train_acc:.4f}\n\t Test Loss: {test_loss:.5f} --  Test"
            f" Accuracy: {test_acc:.4f}"
        )

        if train.checkpoint:
            suf = f"-{train.suffix}" if train.suffix is not None else ""
            run_id = f"checkpoints/{dataset}/{layer}-d_model={model.d_model}-lr={train.lr}-bsz={train.batch_size}{suf}"

            abs_path = os.path.join(os.getcwd(), run_id)

            ckpt_path = checkpoints.save_checkpoint(
                abs_path,
                state,
                epoch,
                keep = train.epochs
            )

        if (test_acc > best_acc) or (test_loss < best_loss):
            if train.checkpoint:
                shutil.copy(ckpt_path, f"{run_id}/best_{epoch}")
                if os.path.exists(f"{run_id}/best_{best_epoch}"):
                    os.remove(f"{run_id}/best_{best_epoch}")

            best_loss, best_acc, best_epoch = test_loss, test_acc, epoch

    log.info(
        f"\tBest Test Loss: {best_loss:.5f} -- Best Test Accuracy:"
        f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
    )


@hydra.main(version_base=None, config_path="",config_name = "config")
def main(cfg: DictConfig) ->None:
    OmegaConf.to_yaml(cfg)

    example_train(**cfg)

if __name__=="__main__":
    main()