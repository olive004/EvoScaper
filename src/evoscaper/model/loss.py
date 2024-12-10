

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
    params, rng, model,
    x: Float[Array, " batch n_interactions"], y: Int[Array, " batch"],
    loss_f,
    use_l2_reg=False, l2_reg_alpha: Float = None,
    use_kl_div=False,
    kl_weight: Float = 1.0,
    **model_call_kwargs
) -> Float[Array, ""]:

    pred_y, mu, logvar = model(
        params, rng, x, return_muvar=True, **model_call_kwargs)
    loss = loss_f(y, pred_y)

    # Add L2 loss
    if use_l2_reg:
        loss += sum(
            l2_loss(w, alpha=l2_reg_alpha)
            for w in jax.tree_util.tree_leaves(params)
        )
    # KL divergence
    if use_kl_div:
        loss += kl_gaussian(mu, logvar).mean() * kl_weight
    return loss


def kl_gaussian(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """ https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_vae.html """
    return 0.5 * jnp.sum(-logvar - 1.0 + jnp.exp(logvar) + jnp.square(mu), axis=-1)


# def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
#     x = jnp.reshape(x, (x.shape[0], -1))
#     logits = jnp.reshape(logits, (logits.shape[0], -1))

#     return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


# def recon_loss(y_true, y_pred):
# 	return jnp.sum(binary_cross_entropy(y_true, y_pred), axis=-1)


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
    threshold=0.1, **model_call_kwargs
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x, **model_call_kwargs)
    return accuracy_regression(pred_y, y, threshold)


@eqx.filter_jit
def accuracy_regression(
    pred_y: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"],
    threshold=0.1
) -> Float[Array, ""]:
    return jnp.mean(jnp.abs(y - pred_y) <= threshold)
