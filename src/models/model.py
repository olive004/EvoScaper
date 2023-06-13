import haiku as hk
import jax
import jax.numpy as jnp
import graphviz

def f(x):
    # return hk.nets.MLP([300, 100, 10])(x)
    return hk.nets.ResNet([28 * 28, 10, 5, 5], 1)(x, is_training=False)


f = hk.transform(f)

rng = jax.random.PRNGKey(42)
x = jnp.ones([8, 28, 28])
params = f.init(rng, x)

dot = hk.experimental.to_dot(f.apply)(params, None, x)
graphviz.Source(dot)
