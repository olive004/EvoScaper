from typing import List
from functools import partial
import json

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import wandb

import pandas as pd


# https://coderzcolumn.com/tutorials/artificial-intelligence/haiku-cnn

class MLP(hk.Module):

    def __init__(self, layer_sizes: List[int], n_head: int, use_categorical: bool):
        super().__init__(name="FCN")
        self.layers = self.create_layers(layer_sizes, n_head, use_categorical)
        
        
    def create_layers(self, layer_sizes: List[int], n_head: int, use_categorical: bool):
        sizes = layer_sizes + [n_head]
        l = []
        for i, s in enumerate(sizes):
            if l:
                l.append(jax.nn.relu)
                if np.mod(i, 2) == 0:
                    l.append(jax.nn.sigmoid)
            # if sj == n_head:
            #     l.append(eqx.nn.Dropout(p=0.4))
            
            # He initialisation
            l.append(
                hk.Linear(s, w_init=hk.initializers.VarianceScaling(scale=2.0))
            )
            
        if use_categorical:
            l.append(jax.nn.log_softmax)
        return l
        

    def __call__(self, x: Float[Array, " num_interactions"], inference: bool = False, seed: int = 0) -> Float[Array, " n_head"]:
        for i, layer in enumerate(self.layers):
            kwargs = {} if not type(layer) == eqx.nn.Dropout else {
                'inference': inference, 'key': jax.random.PRNGKey(seed)}

            x = layer(x, **kwargs)
            
            if inference:
                df = pd.DataFrame(data=np.array(x), columns=['0'])
                # logs[f'emb_{i}_{type(layer)}'] = df
                wandb.log({f'emb_{i}_{type(layer)}': df})
        return x
    
    
def MLP_fn(x, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = MLP(**init_kwargs)
    return model(x, **call_kwargs)