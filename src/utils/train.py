

from functools import partial
import numpy as np
import jax
import optax


def train_step(params, x, y, optimiser_state, model, rng, l2_reg_alpha, optimiser, loss_fn):

    loss, grads = jax.value_and_grad(loss_fn)(
        params, rng, model, x, y, l2_reg_alpha=l2_reg_alpha)

    updates, optimiser_state = optimiser.update(grads, optimiser_state)
    params = optax.apply_updates(params, updates)

    return params, optimiser_state, loss, grads


def eval_step(params, rng, model, x, y, l2_reg_alpha, loss_fn, compute_accuracy):
    """ Return the average of loss and accuracy on validation data """
    loss = loss_fn(params, rng, model, x, y, l2_reg_alpha=l2_reg_alpha)
    acc = compute_accuracy(params, rng, model, x, y)
    return acc, loss


def run_batches(params, model, xy_train, rng, l2_reg_alpha, optimiser, optimiser_state, loss_fn):

    f_train_step = partial(train_step, model=model, rng=rng,
                           l2_reg_alpha=l2_reg_alpha, optimiser=optimiser,
                           loss_fn=loss_fn)

    def f(carry, inp):

        params, optimiser_state = carry[0], carry[1]
        x_batch, y_batch = inp[0], inp[1]

        params, optimiser_state, loss, grads = f_train_step(
            params, x_batch, y_batch, optimiser_state)
        return (params, optimiser_state), (loss, grads)

    # for x_batch, y_batch in xy_train:
    (params, optimiser_state), (train_loss, grads) = jax.lax.scan(
        f, (params, optimiser_state), xy_train)
    return params, optimiser_state, train_loss, grads


def train(params, rng, model, xy_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
          optimiser, optimiser_state,
          l2_reg_alpha: float, epochs: int,
          compute_accuracy,
          loss_fn,
          save_every: int = 50,
          include_params_in_saves: bool = False
          ):

    def f(carry, _):
        params, optimiser_state = carry[0], carry[1]

        params, optimiser_state, train_loss, grads = run_batches(
            params, model, xy_train, rng, l2_reg_alpha, optimiser, optimiser_state, loss_fn)

        val_acc, val_loss = eval_step(
            params, rng, model, x_val, y_val, l2_reg_alpha, loss_fn, compute_accuracy)

        return (params, optimiser_state), (params, grads, train_loss, val_loss, val_acc)

    def do_scan(params, optimiser_state, epochs):
        (params, optimiser_state), (params_stack, grads, train_loss, val_loss, val_acc) = jax.lax.scan(
            f, init=(params, optimiser_state), xs=None, length=epochs)
        return params, train_loss, val_loss, val_acc, params_stack, grads

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

    try:
        params, train_loss, val_loss, val_acc, params_stack, grads = do_scan()
        saves = make_saves(train_loss, val_loss, val_acc, include_params_in_saves, params_stack, grads)
        saves['params'] = params
    except:
        # saves = {}
        # e_max = 50
        # for e in np.arange(0, epochs, e_max):
        #     params, saves_batch = do_scan(params, optimiser_state, e_max)
        #     saves[e] = saves_batch
        #     print(
        #         f'Batch Epoch {e} / {epochs} -\t\t Train loss: {np.mean(saves_batch["train_loss"])}\tVal loss: {np.mean(saves_batch["val_loss"])}\tVal accuracy: {np.mean(saves_batch["val_accuracy"])}')
        #     gc.collect()
        saves = {}
        for e in range(epochs):
            (params, optimiser_state), (params_stack, grads, train_loss,
                                        val_loss, val_acc) = f((params, optimiser_state), None)

            if np.mod(e, save_every) == 0:
                saves[e] = make_saves(train_loss, val_loss, val_acc, include_params_in_saves, params_stack, grads)
                print(
                    f'Epoch {e} / {epochs} -\t\t Train loss: {np.mean(train_loss)}\tVal loss: {val_loss}\tVal accuracy: {val_acc}')

        saves[list(saves.keys())[-1]]['params'] = params
    return params, saves
