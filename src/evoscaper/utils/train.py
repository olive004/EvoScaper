

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
import logging

from evoscaper.utils.dataclasses import TrainingConfig


# @jax.jit
def train_step(params, x, y, cond, optimiser_state, model, rng, config_training: TrainingConfig, optimiser, loss_fn):

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, rng, model, x, y, use_l2_reg=config_training.use_l2_reg, 
        l2_reg_alpha=config_training.l2_reg_alpha, cond=cond,
        use_kl_div=config_training.use_kl_div, kl_weight=config_training.kl_weight, 
        use_contrastive_loss=config_training.use_contrastive_loss,
        temperature=config_training.temperature, threshold_similarity=config_training.threshold_similarity, 
        power_factor_distance=config_training.power_factor_distance)
    
    if config_training.use_grad_clipping:
        grads = clip_gradients(grads, max_norm=1.0)

    updates, optimiser_state = optimiser.update(grads, optimiser_state, params)
    params = optax.apply_updates(params, updates)

    return params, optimiser_state, loss, grads, aux


def eval_step(params, rng, model, x, y, cond, config_training: TrainingConfig, loss_fn, compute_accuracy):
    """ Return the average of loss and accuracy on validation data """
    
    def batch_loss(x, y, cond):
        
        loss, aux = loss_fn(params, rng, model, x, y, use_l2_reg=config_training.use_l2_reg,
                            l2_reg_alpha=config_training.l2_reg_alpha, cond=cond,
                            use_kl_div=config_training.use_kl_div, kl_weight=config_training.kl_weight, 
                            use_contrastive_loss=config_training.use_contrastive_loss,
                            temperature=config_training.temperature, threshold_similarity=config_training.threshold_similarity, 
                            power_factor_distance=config_training.power_factor_distance)
        return loss, aux
    
    loss, aux = jax.vmap(batch_loss)(x, y, cond)
    loss = jnp.mean(loss)
    aux = jax.tree_util.tree_map(lambda x: jnp.mean(x), aux)
    
    pred_y = model(params, rng, x, cond=cond)
    acc = compute_accuracy(pred_y, y)
    return acc, loss, aux


def run_batches(params, model, rng,
                x_batch, y_batch, cond_batch,
                config_training: TrainingConfig, optimiser, optimiser_state, loss_fn):

    f_train_step = partial(train_step, model=model, rng=rng,
                           config_training=config_training, optimiser=optimiser,
                           loss_fn=loss_fn)

    # @jax.jit
    def f(carry, inp):

        params, optimiser_state = carry[0], carry[1]
        x_batch, y_batch, cond_batch = inp[0], inp[1], inp[2]

        params, optimiser_state, loss, grads, aux = f_train_step(
            params, x_batch, y_batch, cond_batch, optimiser_state)
        return (params, optimiser_state), (loss, grads, aux)

    # for x_batch, y_batch in xy_train:
    # (params, optimiser_state), (train_loss, grads) = f((params, optimiser_state), (x_batch, y_batch, cond_batch))
    (params, optimiser_state), (train_loss, grads, aux_loss) = jax.lax.scan(
        f, (params, optimiser_state), (x_batch, y_batch, cond_batch))
    return params, optimiser_state, train_loss, grads, aux_loss


def make_saves(train_loss, val_loss, val_acc, include_params_in_all_saves, params_stack=None, grads=None, aux_loss=None, aux_val_loss=None):
    saves = {
        'train_loss': np.mean(train_loss),
        'val_loss': np.mean(val_loss),
        'val_accuracy': np.mean(val_acc),
    }
    if aux_loss is not None:
        saves['l2_loss'] = np.mean(aux_loss[0]) if aux_loss[0] is not None else None
        saves['kl_loss'] = np.mean(aux_loss[1]) if aux_loss[1] is not None else None
        saves['contrastive_loss'] = np.mean(aux_loss[2]) if aux_loss[2] is not None else None
    if aux_val_loss is not None:
        saves['l2_loss'] = np.mean(aux_val_loss[0]) if aux_val_loss[0] is not None else None
        saves['kl_loss'] = np.mean(aux_val_loss[1]) if aux_val_loss[1] is not None else None
        saves['contrastive_loss'] = np.mean(aux_val_loss[2]) if aux_val_loss[2] is not None else None
    if include_params_in_all_saves:
        saves['params'] = params_stack
        saves['grads'] = grads
    return saves


def clip_gradients(gradients, max_norm):
    """
    Clip gradients to have a maximum norm of max_norm.
    
    Args:
        gradients: PyTree of gradients
        max_norm: float, maximum norm for gradients
        
    Returns:
        PyTree of clipped gradients
    """
    # Calculate global norm across all parameters
    global_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(gradients))
    )
    
    # Calculate clip ratio
    clip_ratio = jnp.minimum(max_norm / (global_norm + 1e-6), 1.0)
    
    # Apply clipping
    clipped_gradients = jax.tree_map(
        lambda x: x * clip_ratio,
        gradients
    )
    
    return clipped_gradients


def early_stopping(val_loss, best_val_loss, val_acc, best_val_acc, epochs_no_improve):
    if (val_loss < best_val_loss) or (val_acc < best_val_acc):
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
        elif (val_acc > best_val_acc):
            best_val_acc = val_acc
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    return epochs_no_improve, best_val_loss


def train(params, rng, model,
          x_train, cond_train, y_train, x_val, cond_val, y_val,
          optimiser, optimiser_state,
          config_training: TrainingConfig, epochs,
          loss_fn, compute_accuracy,
          save_every, include_params_in_all_saves,
          patience: int = 1000):

    best_val_loss = jnp.inf
    best_val_acc = 0
    epochs_no_improve = 0
    info_early_stop = ''
    saves = {}
    for epoch in range(epochs):
        # Shuffle data
        _, rng = jax.random.split(rng)
        perm = jax.random.permutation(rng, len(x_train))
        x_train, y_train, cond_train = x_train[perm], y_train[perm], cond_train[perm]

        def f(carry, _):
            params, optimiser_state = carry[0], carry[1]

            params, optimiser_state, train_loss, grads, aux_loss = run_batches(
                params, model, rng, x_train, y_train, cond_train, config_training, optimiser, optimiser_state, loss_fn)

            val_acc, val_loss, aux_val_loss = eval_step(
                params, rng, model, x_val, y_val, cond_val, config_training, loss_fn, compute_accuracy)

            return (params, optimiser_state), (params, grads, train_loss, val_loss, val_acc, aux_loss, aux_val_loss)
        
        # Run
        (params, optimiser_state), (params_stack, grads, train_loss,
                                    val_loss, val_acc, aux_loss, aux_val_loss) = f((params, optimiser_state), None)

        # Save
        if np.mod(epoch, save_every) == 0:
            saves[epoch] = make_saves(
                train_loss, val_loss, val_acc, include_params_in_all_saves, params_stack, grads, aux_loss, aux_val_loss)
        if np.mod(epoch, 10) == 0:
            logging.info(
                f'Epoch {epoch} / {epochs} -\t\t Train loss: {np.mean(train_loss)}\tVal loss: {val_loss}\tVal accuracy: {val_acc}')

        # Early stopping
        epochs_no_improve, best_val_loss = early_stopping(
            val_loss, best_val_loss, val_acc, best_val_acc, epochs_no_improve)

        # Stop if no improvement or nans
        if (epochs_no_improve > patience) or (np.isnan(np.mean(train_loss)) or np.isnan(val_loss) or np.isnan(val_acc)) or (val_acc > 0.995):
            info_early_stop = f'Early stopping triggered after {epoch+1} epochs:\nTrain loss: {np.mean(train_loss)}\nVal loss: {val_loss}\nVal accuracy: {val_acc}\nEpochs no improvement: {epochs_no_improve}'
            logging.warning(info_early_stop)
            break

    saves[list(saves.keys())[-1]]['params'] = params
    return params, saves, info_early_stop






# from functools import partial
# import numpy as np
# import jax
# import jax.numpy as jnp
# import optax
# import logging


# def eval_step(params, rng, model, x, y, cond, use_l2_reg, l2_reg_alpha, loss_fn, compute_accuracy):
#     """ Return the average of loss and accuracy on validation data """
#     loss, aux = loss_fn(params, rng, model, x, y, use_l2_reg=use_l2_reg,
#                         l2_reg_alpha=l2_reg_alpha, cond=cond)
#     pred_y = model(params, rng, x, cond=cond)
#     acc = compute_accuracy(pred_y, y)
#     return acc, loss, aux


# def run_batches(params, model, rng,
#                 x_batch, y_batch, cond_batch,
#                 use_l2_reg, l2_reg_alpha, optimiser, optimiser_state, loss_fn):

#     # f_train_step = partial(train_step, model=model, rng=rng,
#     #                        use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, optimiser=optimiser,
#     #                        loss_fn=loss_fn)

#     @jax.jit
#     def train_step(params, x, y, cond, optimiser_state):

#         (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
#             params, rng, model, x, y, use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, cond=cond)

#         updates, optimiser_state = optimiser.update(grads, optimiser_state)
#         params = optax.apply_updates(params, updates)

#         return params, optimiser_state, loss, grads, aux

#     @jax.jit
#     def f(carry, inp):

#         params, optimiser_state = carry[0], carry[1]
#         x_batch, y_batch, cond_batch = inp[0], inp[1], inp[2]

#         params, optimiser_state, loss, grads, aux = train_step(
#             params, x_batch, y_batch, cond_batch, optimiser_state)
#         return (params, optimiser_state), (loss, grads, aux)

#     (params, optimiser_state), (train_loss, grads, aux_loss) = jax.lax.scan(
#         f, (params, optimiser_state), (x_batch, y_batch, cond_batch))
#     return params, optimiser_state, train_loss, grads, aux_loss


# def make_saves(train_loss, val_loss, val_acc, include_params_in_all_saves, params_stack=None, grads=None, aux_loss=None, aux_val_loss=None):
#     saves = {
#         'train_loss': np.mean(train_loss),
#         'val_loss': np.mean(val_loss),
#         'val_accuracy': np.mean(val_acc),
#     }
#     if aux_loss is not None:
#         saves['l2_loss'] = np.mean(aux_loss[0]) if aux_loss[0] is not None else None
#         saves['kl_loss'] = np.mean(aux_loss[1]) if aux_loss[1] is not None else None
#     if aux_val_loss is not None:
#         saves['l2_loss'] = np.mean(aux_val_loss[0]) if aux_val_loss[0] is not None else None
#         saves['kl_loss'] = np.mean(aux_val_loss[1]) if aux_val_loss[1] is not None else None
#     if include_params_in_all_saves:
#         saves['params'] = params_stack
#         saves['grads'] = grads
#     return saves


# def early_stopping(val_loss, best_val_loss, epochs_no_improve):
#     if val_loss > best_val_loss:
#         best_val_loss = val_loss
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1
#     return epochs_no_improve, best_val_loss


# def train(params, rng, model,
#           x_train, cond_train, y_train, x_val, cond_val, y_val,
#           optimiser, optimiser_state,
#           use_l2_reg, l2_reg_alpha, epochs,
#           loss_fn, compute_accuracy,
#           save_every, include_params_in_all_saves,
#           patience: int = 1000):

#     best_val_loss = jnp.inf
#     epochs_no_improve = 0
#     saves = {}
#     for epoch in range(epochs):
#         # Shuffle data
#         _, rng = jax.random.split(rng)
#         perm = jax.random.permutation(rng, len(x_train))
#         x_train, y_train, cond_train = x_train[perm], y_train[perm], cond_train[perm]

#         def f(carry, _):
#             params, optimiser_state = carry[0], carry[1]

#             params, optimiser_state, train_loss, grads, aux_loss = run_batches(
#                 params, model, rng, x_train, y_train, cond_train, use_l2_reg, l2_reg_alpha, optimiser, optimiser_state, loss_fn)

#             val_acc, val_loss, aux_val_loss = eval_step(
#                 params, rng, model, x_val, y_val, cond_val, use_l2_reg, l2_reg_alpha, loss_fn, compute_accuracy)

#             return (params, optimiser_state), (params, grads, train_loss, val_loss, val_acc, aux_loss, aux_val_loss)
        
#         # Run
#         (params, optimiser_state), (params_stack, grads, train_loss,
#                                     val_loss, val_acc, aux_loss, aux_val_loss) = f((params, optimiser_state), None)

#         # Save
#         if np.mod(epoch, save_every) == 0:
#             saves[epoch] = make_saves(
#                 train_loss, val_loss, val_acc, include_params_in_all_saves, params_stack, grads, aux_loss, aux_val_loss)
#             logging.info(
#                 f'Epoch {epoch} / {epochs} -\t\t Train loss: {np.mean(train_loss)}\tVal loss: {val_loss}\tVal accuracy: {val_acc}')

#         # Early stopping
#         epochs_no_improve, best_val_loss = early_stopping(
#             val_loss, best_val_loss, epochs_no_improve)

#         # Stop if no improvement or nans
#         if (epochs_no_improve > patience) or (np.isnan(np.mean(train_loss)) or np.isnan(val_loss) or np.isnan(val_acc)):
#             logging.info(f'Early stopping triggered after {epoch+1} epochs')
#             break

#     saves[list(saves.keys())[-1]]['params'] = params
#     return params, saves
