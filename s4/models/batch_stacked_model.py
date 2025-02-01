from flax import linen as nn
import jax.numpy as np
import jax


class SequenceBlock(nn.Module):
    layer_cls: nn.module
    layer: dict                     # A dictionary containing hyperparameters for the layer_cls
    dropout: float
    d_model: int                    # Dimensionality of output features
    prenorm: bool = True            # Whether to apply LayerNorm before sequential layer
    glu: bool = True                # Gate Linear Unit for additional processing
    training: bool = True
    decode: bool = False            # CNN or RNN mode

    def setup(self):
        self.seq = self.layer_cls(**self.layer, decode=self.decode)
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.d_model)

        if self.glu:
            self.out2 = nn.Dense(self.d_model)

        self.drop = nn.Dropout(self.dropout, broadcast_dims=[
                               0], deterministic=not self.training)

    def __call__(self, x):
        skip = x

        if self.prenorm:
            x = self.norm(x)

        x = self.seq(x)
        x = self.drop(nn.gelu(x))

        if self.glu:
            x = self.out(x) * jax.nn.sigmoid(self.out2(x))
        else:
            x = self.out(x)

        x = skip + self.drop(x)

        if not self.prenorm:
            x = self.norm(x)

        return x


class Embedding(nn.Embed):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        y = nn.Embed(self.num_embeddings, self.features)(x[..., 0])
        return np.where(x > 0, y, 0.0)


class StackedModel(nn.Module):
    layer_cls: nn.Module
    layer: dict
    d_output: int
    d_model: int
    n_layers: int
    prenorm: bool = True
    dropout: float = 0.0
    embedding: bool = False
    classification: bool = False
    training: bool = True
    decode: bool = False

    def setup(self):

        if self.embedding:
            self.encoder = Embedding(self.d_output, self.d_model)
        else:
            self.encoder = nn.Dense(self.d_model)

        self.decoder = nn.Dense(self.d_output)
        self.layers = [
            SequenceBlock(
                layer_cls=self.layer_cls,
                layer=self.layer,
                dropout=self.dropout,
                d_model=self.d_model,
                prenorm=self.prenorm,
                training=self.training,
                decode=self.decode
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x):
        if not self.classification:
            if not self.embedding:
                # When we are working on Image Data we divide by 255.
                x = x / 255.0
            if not self.decode:
                x = np.pad(x[:-1], [(1, 0), (0, 0)])

        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x)

        if self.classification:
            x = np.mean(x, axis=0)

        x = self.decoder(x)

        return nn.log_softmax(x, axis=-1)


BatchStackedModel = nn.vmap(
    StackedModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
)
