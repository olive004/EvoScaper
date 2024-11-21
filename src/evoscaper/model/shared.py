

import jax.numpy as jnp


def model_fn(x, model, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = model(**init_kwargs)
    return model(x, **call_kwargs)


def arrayise(d):
    for k, v in d.items():
        if type(v) == dict:
            for kk, vv in v.items():
                d[k][kk] = jnp.array(vv)
    return d