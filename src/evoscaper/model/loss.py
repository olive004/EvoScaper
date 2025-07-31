

from typing import Tuple
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
    x: jnp.ndarray, y: jnp.ndarray,
    loss_f,
    use_l2_reg=False, l2_reg_alpha: Float = None,
    use_kl_div=False,
    kl_weight: Float = 1.0,
    use_contrastive_loss: bool = False,
    temperature: float = 1.0,
    threshold_similarity=0.9,
    power_factor_distance=3,
    **model_call_kwargs
) -> Float[Array, ""]:

    pred_y, mu, logvar, h = model(
        params, rng, x, return_all=True, **model_call_kwargs)
    loss = loss_f(y, pred_y)

    # Add L2 loss
    loss_l2 = None
    if use_l2_reg:
        loss_l2 = sum(
            l2_loss(w, alpha=l2_reg_alpha)
            for w in jax.tree_util.tree_leaves(params)
        )
        loss += loss_l2

    # KL divergence
    loss_kl = None
    if use_kl_div:
        loss_kl = kl_gaussian(mu, logvar).mean() * kl_weight
        loss += loss_kl

    # Contrastive loss
    loss_cl = None
    if use_contrastive_loss:
        loss_cl = contrastive_loss_fn(
            model_call_kwargs.get('cond'), h, threshold_similarity, temperature, power_factor_distance)
        loss += loss_cl
    return loss, (loss_l2, loss_kl, loss_cl)


def kl_gaussian(mu: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """ https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_vae.html """
    return 0.5 * jnp.sum(-logvar - 1.0 + jnp.exp(logvar) + jnp.square(mu), axis=-1)


# def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
#     x = jnp.reshape(x, (x.shape[0], -1))
#     logits = jnp.reshape(logits, (logits.shape[0], -1))

#     return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


# def recon_loss(y_true, y_pred):
# 	return jnp.sum(binary_cross_entropy(y_true, y_pred), axis=-1)


def l1_norm(x, y):
    return jnp.sum(jnp.abs(x - y), axis=1)


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
    rtol=1e-3, atol=1e-5, **model_call_kwargs
) -> Float[Array, ""]:
    pred_y = model.apply(params, rng, x, **model_call_kwargs)
    return accuracy_regression(pred_y, y, rtol=rtol, atol=atol)


@eqx.filter_jit
def accuracy_regression_exact(
    pred_y: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"],
    rtol=1e-3, atol=1e-5
) -> Float[Array, ""]:
    return jnp.mean(jnp.isclose(pred_y, y, rtol=rtol, atol=atol))


@eqx.filter_jit
def accuracy_regression(
    pred_y: Float[Array, "batch num_interactions"], y: Int[Array, " batch n_head"],
    threshold=0.1, **kwargs
) -> Float[Array, ""]:
    return jnp.mean(jnp.abs(y - pred_y) <= threshold)


@eqx.filter_jit
def normalise_embeddings(embeddings: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize embeddings to unit length.
    """
    norms = jnp.sqrt(jnp.sum(embeddings ** 2, axis=1, keepdims=True))
    return embeddings / (norms + 1e-8)


def contrastive_loss_fn(cond: jnp.ndarray,
                        h: jnp.ndarray,
                        threshold_similarity: float = 0.9,
                        temperature: float = 1.0,
                        power_factor_distance: int = 3,
                        distance_metric: str = 'dot') -> jnp.ndarray:
    """ Compute contrastive loss for self-supervised learning.
    Works for one hot encoding and continuous labels.
    Scaled similarities and the distance between labels will be between 0 and 1.
    Encodings for which the label is above the threshold of similarity `threshold_similarity` 
    will decrease loss (these are the "positive" samples), while those within the threshold 
    will increase the loss (these are the "negative" samples).
    Temperature increase will lower the loss, while temperature closer to 0 will increase the loss.
    """

    cond_norm = normalise_embeddings(cond)
    h_norm = normalise_embeddings(h)

    cond_dist = contrastive_distance_labels(
        cond_norm, distance_metric, power_factor_distance)

    scaled_similarities = jnp.dot(h_norm, h_norm.T) / temperature

    mask_self_samples = jnp.where(
        jnp.eye(scaled_similarities.shape[0]) == 0, 1, 0)

    loss = jnp.mean(- scaled_similarities * (cond_dist -
                    threshold_similarity) * mask_self_samples)

    return loss


def contrastive_distance_labels(c, distance_metric='dot', power_factor_distance=3):

    if distance_metric == 'dot':
        # Min -1, max 1
        distance = jnp.power(jnp.dot(c, c.T), power_factor_distance)

    elif distance_metric == 'l1_norm':
        distance = l1_norm(c[:, :, None], c.T)

    return distance
