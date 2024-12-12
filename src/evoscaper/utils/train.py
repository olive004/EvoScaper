

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
import logging


def train_step(params, x, y, cond, optimiser_state, model, rng, use_l2_reg, l2_reg_alpha, optimiser, loss_fn):

    loss_fn(
        params, rng, model, x, y, use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, cond=cond)
    loss, grads = jax.value_and_grad(loss_fn)(
        params, rng, model, x, y, use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, cond=cond)

    updates, optimiser_state = optimiser.update(grads, optimiser_state)
    params = optax.apply_updates(params, updates)

    return params, optimiser_state, loss, grads


def eval_step(params, rng, model, x, y, cond, use_l2_reg, l2_reg_alpha, loss_fn, compute_accuracy):
    """ Return the average of loss and accuracy on validation data """
    loss = loss_fn(params, rng, model, x, y, use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, cond=cond)
    pred_y = model(params, rng, x, cond=cond)
    acc = compute_accuracy(pred_y, y)
    return acc, loss


def run_batches(params, model, rng,
                x_batch, y_batch, cond_batch,
                use_l2_reg, l2_reg_alpha, optimiser, optimiser_state, loss_fn):

    f_train_step = partial(train_step, model=model, rng=rng,
                           use_l2_reg=use_l2_reg, l2_reg_alpha=l2_reg_alpha, optimiser=optimiser,
                           loss_fn=loss_fn)

    def f(carry, inp):

        params, optimiser_state = carry[0], carry[1]
        x_batch, y_batch, cond_batch = inp[0], inp[1], inp[2]

        params, optimiser_state, loss, grads = f_train_step(
            params, x_batch, y_batch, cond_batch, optimiser_state)
        return (params, optimiser_state), (loss, grads)

    # for x_batch, y_batch in xy_train:
    # (params, optimiser_state), (train_loss, grads) = f((params, optimiser_state), (x_batch, y_batch, cond_batch))
    (params, optimiser_state), (train_loss, grads) = jax.lax.scan(
        f, (params, optimiser_state), (x_batch, y_batch, cond_batch))
    return params, optimiser_state, train_loss, grads


def make_saves(train_loss, val_loss, val_acc, include_params_in_saves, params_stack=None, grads=None):
    saves = {
        'train_loss': np.mean(train_loss),
        'val_loss': np.mean(val_loss),
        'val_accuracy': np.mean(val_acc)
    }
    if include_params_in_saves:
        saves['params'] = params_stack
        saves['grads'] = grads
    return saves


def early_stopping(val_loss, best_val_loss, epochs_no_improve):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    return epochs_no_improve, best_val_loss



def train(params, rng, model,
          x_train, cond_train, y_train, x_val, cond_val, y_val,
          optimiser, optimiser_state,
          use_l2_reg, l2_reg_alpha, epochs,
          loss_fn, compute_accuracy,
          save_every, include_params_in_saves,
          patience: int = 1000):

    def f(carry, _):
        params, optimiser_state = carry[0], carry[1]

        params, optimiser_state, train_loss, grads = run_batches(
            params, model, rng, x_train, y_train, cond_train, use_l2_reg, l2_reg_alpha, optimiser, optimiser_state, loss_fn)

        val_acc, val_loss = eval_step(
            params, rng, model, x_val, y_val, cond_val, use_l2_reg, l2_reg_alpha, loss_fn, compute_accuracy)

        return (params, optimiser_state), (params, grads, train_loss, val_loss, val_acc)

    best_val_loss = jnp.inf
    epochs_no_improve = 0
    saves = {}
    for epoch in range(epochs):
        # Run 
        (params, optimiser_state), (params_stack, grads, train_loss,
                                    val_loss, val_acc) = f((params, optimiser_state), None)

        # Save
        if np.mod(epoch, save_every) == 0:
            saves[epoch] = make_saves(
                train_loss, val_loss, val_acc, include_params_in_saves, params_stack, grads)
            logging.info(
                f'Epoch {epoch} / {epochs} -\t\t Train loss: {np.mean(train_loss)}\tVal loss: {val_loss}\tVal accuracy: {val_acc}')
            
        # Early stopping 
        epochs_no_improve, best_val_loss = early_stopping(
            val_loss, best_val_loss, epochs_no_improve)
        
        # Stop if no improvement
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break
        
    saves[list(saves.keys())[-1]]['params'] = params
    return params, saves
