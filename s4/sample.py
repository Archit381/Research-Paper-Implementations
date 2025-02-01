import jax
from s4.models.s4_model import S4Layer
from s4.models.batch_stacked_model import BatchStackedModel
import matplotlib.pyplot as plt
from flax.training import checkpoints
import jax.numpy as np
import logging
from rich.logging import RichHandler
from flax.core import unfreeze

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

log = logging.getLogger("rich")


def sample(model, params, prime, cache, x, start, end, rng):
    def loop(i, cur):
        x, rng, cache = cur
        r, rng = jax.random.split(rng)

        out, vars = model.apply(
            {"params": params, "prime": prime, "cache": cache},
            x[:, np.arange(1,2)*i],
            mutable = ["cache"]
        )

        def update(x, out):
            p = jax.random.categorical(r, out[0])
            x = x.at[i+1, 0].set(p)
            return x
        
        x = jax.vmap(update)(x, out)
        return x, rng, unfreeze(vars['cache'])

    return jax.lax.fori_loop(start, end, jax.jit(loop), (x, rng, cache))[0]

def init_recurrence(model, params, init_x, rng):
    variables = model.init(rng, init_x)

    vars = {
        "params": unfreeze(variables["params"]),
        "cache": unfreeze(variables['cache']),
        "prime": unfreeze(variables['prime']),
    }

    vars["params"].update(params)

    _, prime_vars = model.apply(vars, init_x, mutable=['prime'])
    return vars['params'], prime_vars['prime'], vars['cache']

def sample_checkpoint(path, model, length, rng):

    start = np.zeros((1, length, 1), dtype=int)
    log.info("[*] Initializing from checkpoint %s" % path)

    state = checkpoints.restore_checkpoint(path, None)

    assert 'params' in state

    params, prime, cache = init_recurrence(model, state['params'], start, rng)

    return sample(model, params, prime, cache, start, 0, length-1, rng)


def main():

    rng = jax.random.PRNGKey(1)

    layer_args = {}
    layer_args["N"] = 64
    layer_args["l_max"] = 784

    layer_cls = S4Layer

    model = BatchStackedModel(
        layer_cls = layer_cls,
        layer = layer_args,
        d_output = 256,
        d_model = 128,
        n_layers = 6,
        prenorm = False,
        classification = False,
        decode = True,
        training=False
    )

    MNIST_LEN = 784
    default_train_path = r"C:\Users\Archit\Desktop\random_stuff\Research-Paper-Implementations\checkpoints\mnist\s4\checkpoint_0"

    out = sample_checkpoint(
        default_train_path, model, MNIST_LEN, rng
    )

    plt.imshow(out.reshape(28, 28))
    plt.savefig("s4/sample.png")

if __name__=="__main__":
    main()
