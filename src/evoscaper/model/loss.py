

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
    x: Float[Array, " batch n_interactions"], y: Int[Array, " batch"],
    loss_f,
    use_l2_reg=False, l2_reg_alpha: Float = None,
    use_kl_div=False,
    kl_weight: Float = 1.0,
    use_contrastive_loss: bool = False,
    temperature: float = 1.0,
    batch_size_max_contloss = 64,
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
    if use_contrastive_loss:
        implement_contrastive_loss(h, y, temperature, batch_size_max_contloss)
    return loss, (loss_l2, loss_kl)


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
def normalize_embeddings(embeddings: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize embeddings to unit length.
    """
    norms = jnp.sqrt(jnp.sum(embeddings ** 2, axis=1, keepdims=True))
    return embeddings / (norms + 1e-8)


@eqx.filter_jit
def contrastive_loss_fn(anchor: jnp.ndarray,
                       positive: jnp.ndarray,
                       temperature: float = 0.1,
                       normalize: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute contrastive loss for self-supervised learning.
    
    Args:
        anchor: Anchor embeddings of shape (batch_size, embedding_dim)
        positive: Positive embeddings of shape (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling similarities
        normalize: Whether to L2 normalize embeddings
    
    Returns:
        Tuple of (loss, similarities matrix)
    """
    batch_size = anchor.shape[0]
    
    # Normalize embeddings if requested
    if normalize:
        anchor = normalize_embeddings(anchor)
        positive = normalize_embeddings(positive)
    
    # Compute similarities between all possible pairs
    anchor_dot_positive = jnp.dot(anchor, positive.T)  # (batch_size, batch_size)
    
    # Scale similarities by temperature
    scaled_similarities = anchor_dot_positive / temperature
    
    # For each anchor, the positive example is on the diagonal
    labels = jnp.eye(batch_size)
    
    # Compute cross entropy loss
    log_softmax = jax.nn.log_softmax(scaled_similarities, axis=1)
    loss = -jnp.sum(labels * log_softmax) / batch_size
    
    return loss, scaled_similarities


def contrastive_distance_labels(y, distance_metric='dot'):
    
    if distance_metric == 'dot':
        yy = normalize_embeddings(y)
        distance = jnp.power(jnp.dot(yy, yy.T), 2)

    elif distance_metric == 'l1_norm':
        distance = l1_norm(y[:, :, None], y.T)
        
    return distance


def implement_contrastive_loss(h, y, temperature, batch_size_max_contloss, similarity_threshold = 0.9):
    # batch_size = h.shape[0]
    # if batch_size > batch_size_max_contloss:
    #     h.reshape()
        
    def implement_batch(h, y):
        y_distances = contrastive_distance_labels(y)
        
        loss = 0
        for y_dist in y_distances:
            anchor = h[y_dist < similarity_threshold]
            positive = h[y_dist > similarity_threshold]
            contrastive_loss_fn(anchor, positive, temperature)
        
    for h_i, y_i in zip(h, y):
        implement_batch()
