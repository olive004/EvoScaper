from typing import List
import numpy as np
import haiku as hk
import jax
import equinox as eqx
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import wandb
from collections.abc import Iterable
from typing import Callable, Optional
import pandas as pd


# https://coderzcolumn.com/tutorials/artificial-intelligence/haiku-cnn

class MLPWithActivation(hk.Module):

    def __init__(self,
                 output_sizes: Iterable[int],
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 with_bias: bool = True,
                 activation: Optional[Callable[[
                     jax.Array], jax.Array]] = None,
                 activation_final: Optional[Callable[[
                     jax.Array], jax.Array]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activation_final = activation_final
        self.layers = self.create_layers(output_sizes)

    def create_layers(self, output_sizes: List[int]):
        l = []
        for i, output_size in enumerate(output_sizes):

            l.append(hk.Linear(output_size=output_size,
                               w_init=self.w_init,
                               b_init=self.b_init,
                               with_bias=self.with_bias,
                               name="linear_%d" % i))

        if self.activation_final is not None:
            l.append(self.activation_final)
        return l

    def __call__(self, x: Float[Array, " num_interactions"], inference: bool = False, seed: int = 0, logging: bool = True) -> Float[Array, " n_head"]:
        for i, layer in enumerate(self.layers):
            kwargs = {} if not type(layer) == eqx.nn.Dropout else {
                'inference': inference, 'key': jax.random.PRNGKey(seed)}

            x = layer(x, **kwargs)
            if (self.activation is not None) and (i < len(self.layers) - 1):
                x = self.activation(x)

            if inference and logging:
                df = pd.DataFrame(data=np.array(x), columns=['0'])
                # logs[f'emb_{i}_{type(layer)}'] = df
                wandb.log({f'emb_{i}_{type(layer)}': df})

        if self.activation_final is not None:
            x = self.activation_final(x)
            
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
    model = MLPWithActivation(**init_kwargs)
    return model(x, **call_kwargs)
