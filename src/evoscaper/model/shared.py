

import jax.numpy as jnp
import haiku as hk


def model_fn(x, model, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = model(**init_kwargs)
    return model(x, **call_kwargs)


def arrayise(d):
    for k, v in d.items():
        if type(v) == dict:
            for kk, vv in v.items():
                d[k][kk] = jnp.array(vv)
    return d


def get_initialiser(init):
    if init == 'HeNormal':
        return hk.initializers.VarianceScaling(scale=2.0)
    elif init == 'RandomNormal':
        return hk.initializers.RandomNormal()
    else:
        raise ValueError(f'Invalid initialiser: {init}')