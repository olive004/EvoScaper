

from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping
from sklearn.metrics import r2_score
import jax.numpy as jnp
import jax
import optax  # https://github.com/deepmind/optax
import equinox as eqx
import haiku as hk


def loss_fn(
    params, rng,
    model: hk.Module, x: Float[Array, " batch n_interactions"], y: Int[Array, " batch"],
    use_l2_reg=False, l2_reg_alpha: Float = None,
    loss_type: str = 'categorical'
) -> Float[Array, ""]:

    pred_y = model.apply(params, rng, x)
    if loss_type == 'categorical':
        loss = cross_entropy(y, pred_y, num_classes=pred_y.shape[-1]) / len(x)
    else:
        loss = mse_loss(y, pred_y.reshape(y.shape))

    # Add L2 loss
    if use_l2_reg:
        loss += sum(
            l2_loss(w, alpha=l2_reg_alpha)
            for w in jax.tree_util.tree_leaves(params)
        )
    return loss


def loss_wrapper(
    params, rng,
    model_f, x: Float[Array, " batch n_interactions"], y: Int[Array, " batch"],
    loss_f,
    use_l2_reg=False, l2_reg_alpha: Float = None, 
    **model_call_kwargs
) -> Float[Array, ""]:
    
    pred_y = model_f(params, rng, x, **model_call_kwargs)
    loss = loss_f(y, pred_y)

    # Add L2 loss
    if use_l2_reg:
        loss += sum(
            l2_loss(w, alpha=l2_reg_alpha)
            for w in jax.tree_util.tree_leaves(params)
        )
    return loss


def l2_loss(weights, alpha):
    return alpha * (weights ** 2).mean()


def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"], num_classes: int):
    one_hot_actual = jax.nn.one_hot(y, num_classes=num_classes)
    return optax.softmax_cross_entropy(pred_y, one_hot_actual).sum()


def mse_loss(y, pred_y):
    return jnp.mean(jnp.square(pred_y - y))


def update_params(optimiser, params, grads):
    updates, optimizer_state = optimiser.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params


@eqx.filter_jit
def compute_accuracy_categorical(
    params, rng, model: hk.Module, x: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"]
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


@eqx.filter_jit
def compute_accuracy_regression(
    params, rng, model: hk.Module, x: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"],
    threshold=0.1
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x)
    return jnp.mean(jnp.abs(y - pred_y) <= threshold)
