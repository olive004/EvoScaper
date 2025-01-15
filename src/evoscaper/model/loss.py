

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
def contrastive_loss(z, c, temperature=0.5):
    """
    Compute contrastive loss between latent representations and conditions

    Args:
        z: Batch of latent representations [batch_size, latent_dim]
        c: Batch of conditions [batch_size, condition_dim]
        temperature (float): Temperature parameter for scaling similarities
    """
    batch_size = z.shape[0]


    # Normalize embeddings
    z_norm = F.normalize(z, dim=1)
    c_norm = F.normalize(c, dim=1)

    # Compute similarities
    similarity_matrix = torch.matmul(z_norm, c_norm.T) / temperature

    # Positive pairs are on the diagonal
    positives = torch.diag(similarity_matrix)

    # All other pairs are negatives
    negatives = similarity_matrix.view(-1)

    # Create labels: 1 for positive pairs, 0 for negative pairs
    labels = torch.zeros_like(negatives)
    labels[torch.arange(0, batch_size * batch_size, batch_size + 1)] = 1

    # Compute NCE loss
    nce_loss = F.binary_cross_entropy_with_logits(negatives, labels)

    return nce_loss
