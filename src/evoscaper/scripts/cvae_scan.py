

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from datetime import datetime

from evoscaper.model.vae import sample_z, VAE_fn
from evoscaper.model.shared import arrayise, get_activation_fn
from evoscaper.model.loss import loss_wrapper, compute_accuracy_regression, mse_loss, accuracy_regression
from evoscaper.utils.math import convert_to_scientific_exponent
from evoscaper.utils.optimiser import make_optimiser
from evoscaper.utils.train import train
from evoscaper.utils.dataclasses import NormalizationSettings, ModelConfig
from evoscaper.utils.dataset import init_data


def init_model(rng, x, cond, model_settings: ModelConfig):
    # rng = jax.random.PRNGKey(model_settings.seed)

    model_fn = partial(VAE_fn, enc_layers=model_settings.enc_layers, dec_layers=model_settings.dec_layers,
                       decoder_head=model_settings.decoder_head, HIDDEN_SIZE=model_settings.hidden_size,
                       decoder_activation_final=jax.nn.sigmoid if model_settings.use_sigmoid_decoder else jax.nn.leaky_relu,
                       enc_init=model_settings.enc_init, dec_init=model_settings.dec_init,
                       activation=get_activation_fn(model_settings.activation))
    model_t = hk.multi_transform(model_fn)
    if model_settings.init_model_with_random:
        dummy_x = jax.random.normal(rng, x.shape)
        dummy_cond = jax.random.normal(rng, cond.shape)
        params = model_t.init(rng, dummy_x, dummy_cond, deterministic=False)
    params = model_t.init(rng, x, cond, deterministic=False)
    encoder, decoder, model, h2mu, h2logvar, reparam = model_t.apply
    return params, encoder, decoder, model, h2mu, h2logvar, reparam


def init_optimiser(x, learning_rate_sched, learning_rate, epochs, l2_reg_alpha, use_warmup, warmup_epochs, n_batches):
    optimiser = make_optimiser(learning_rate_sched, learning_rate,
                               epochs, l2_reg_alpha, use_warmup, warmup_epochs, n_batches)
    optimiser_state = optimiser.init(x)
    return optimiser, optimiser_state


def make_training_data(x, cond, y, train_split, n_batches, batch_size):
    def f_reshape(i): return i.reshape(n_batches, batch_size, i.shape[-1])

    x = f_reshape(x)
    y = f_reshape(x)
    cond = f_reshape(cond)

    x_train, cond_train, y_train = x[:int(train_split * n_batches)], cond[:int(
        train_split * n_batches)], y[:int(train_split * n_batches)]
    x_val, cond_val, y_val = x[int(train_split * n_batches):], cond[int(
        train_split * n_batches):], y[int(train_split * n_batches):]

    return x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val


def main(hpos):

    rng = jax.random.PRNGKey(hpos['seed'])
    rng_model = jax.random.PRNGKey(hpos['seed_arch'])

    (df, x, cond, TOTAL_DS, N_BATCHES, x_datanormaliser, x_methods_preprocessing,
     y_datanormaliser, y_methods_preprocessing) = init_data(hpos)

    model_settings = ModelConfig(
        decoder_head=x.shape[-1],
        activation=get_activation_fn(ACTIVATION),
        use_sigmoid_decoder=USE_SIGMOID_DECODER
    )
    params, encoder, decoder, model, h2mu, h2logvar, reparam = init_model(
        rng_model, x, cond, model_settings)
    x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val = make_training_data(
        x, cond, y, TRAIN_SPLIT, N_BATCHES, BATCH_SIZE)

    tstart = datetime.now()
    params, saves = train(params, rng, model,
                          x_train, cond_train, y_train, x_val, cond_val, y_val,
                          optimiser, optimiser_state,
                          use_l2_reg=USE_L2_REG, l2_reg_alpha=L2_REG_ALPHA, epochs=EPOCHS,
                          loss_fn=loss_fn, compute_accuracy=accuracy_regression,
                          save_every=PRINT_EVERY, include_params_in_saves=False)

    print(datetime.now() - tstart)
