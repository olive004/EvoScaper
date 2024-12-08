from typing import List
import numpy as np
import haiku as hk
import jax
import equinox as eqx
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import wandb

import pandas as pd


# https://coderzcolumn.com/tutorials/artificial-intelligence/haiku-cnn

class MLP(hk.Module):

    def __init__(self, layer_sizes: List[int], n_head: int, use_categorical: bool, name='MLP'):
        super().__init__(name=name)
        self.layers = self.create_layers(layer_sizes, n_head, use_categorical)

    def create_layers(self, layer_sizes: List[int], n_head: int, use_categorical: bool):
        sizes = layer_sizes + [n_head]
        l = []
        for i, s in enumerate(sizes):
            # if l:
            #     l.append(jax.nn.relu)
            #     if np.mod(i, 2) == 0:
            #         l.append(jax.nn.sigmoid)
            # if sj == n_head:
            #     l.append(eqx.nn.Dropout(p=0.4))

            # He initialisation
            l.append(
                hk.Linear(s, w_init=hk.initializers.VarianceScaling(scale=2.0))
            )
            l.append(jax.nn.leaky_relu)

        if use_categorical:
            l.append(jax.nn.log_softmax)
        return l

    def __call__(self, x: Float[Array, " num_interactions"], inference: bool = False, seed: int = 0, logging: bool = True) -> Float[Array, " n_head"]:
        for i, layer in enumerate(self.layers):
            kwargs = {} if not type(layer) == eqx.nn.Dropout else {
                'inference': inference, 'key': jax.random.PRNGKey(seed)}

            x = layer(x, **kwargs)

            if inference and logging:
                df = pd.DataFrame(data=np.array(x), columns=['0'])
                # logs[f'emb_{i}_{type(layer)}'] = df
                wandb.log({f'emb_{i}_{type(layer)}': df})

        # def f(carry, layer):
        #     x, i = carry
        #     kwargs = {} if not type(layer) == eqx.nn.Dropout else {
        #         'inference': inference, 'key': jax.random.PRNGKey(seed)}
        #     x = layer(x, **kwargs)
        #     if inference and logging:
        #         df = pd.DataFrame(data=np.array(x), columns=['0'])
        #         # logs[f'emb_{i}_{type(layer)}'] = df
        #         i += 1
        #         wandb.log({f'emb_{i}_{type(layer)}': df})
        #     return (x, i), None

        # (x, _i), _ = jax.lax.scan(f, (x, 0), self.layers)
        return x


def MLP_fn(x, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = MLP(**init_kwargs)
    return model(x, **call_kwargs)