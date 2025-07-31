

from functools import partial
import jax
import os
import pandas as pd
import haiku as hk
from evoscaper.model.loss import loss_wrapper, mse_loss, accuracy_regression
from evoscaper.model.shared import get_activation_fn
from evoscaper.model.vae import VAE_fn
from evoscaper.utils.dataclasses import ModelConfig
from evoscaper.utils.dataset import init_data, make_training_data, load_data
from evoscaper.utils.tuning import make_configs_initial, make_config_model


def init_model(rng, x, cond, config_model: ModelConfig):

    model_fn = partial(VAE_fn, enc_layers=config_model.enc_layers, dec_layers=config_model.dec_layers,
                       decoder_head=config_model.decoder_head, HIDDEN_SIZE=config_model.hidden_size,
                       decoder_activation_final=jax.nn.sigmoid if config_model.use_sigmoid_decoder else jax.nn.leaky_relu,
                       enc_init=config_model.enc_init, dec_init=config_model.dec_init,
                       activation=get_activation_fn(config_model.activation))
    model_t = hk.multi_transform(model_fn)
    if config_model.init_model_with_random:
        dummy_x = jax.random.normal(rng, x.shape)
        dummy_cond = jax.random.normal(rng, cond.shape)
        params = model_t.init(rng, dummy_x, dummy_cond, deterministic=False)
    params = model_t.init(rng, x, cond, deterministic=False)
    encoder, decoder, model, h2mu, h2logvar, reparam = model_t.apply
    return params, encoder, decoder, model, h2mu, h2logvar, reparam


def init_from_hpos(hpos: pd.Series):
    rng = jax.random.PRNGKey(int(hpos['seed_train']))
    rng_model = jax.random.PRNGKey(int(hpos['seed_arch']))
    rng_dataset = jax.random.PRNGKey(int(hpos['seed_dataset']))

    # Configs + data
    (config_norm_x, config_norm_y, config_filter, config_optimisation,
     config_dataset, config_training) = make_configs_initial(hpos)
    data, x_cols = load_data(config_dataset)

    # Init data
    (df, x, cond, total_ds, n_batches, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing,
     y_datanormaliser, y_methods_preprocessing) = init_data(
         data, x_cols, config_dataset.objective_col, config_dataset.output_species, config_dataset.total_ds_max,
         config_training.batch_size, rng_dataset, config_norm_x, config_norm_y, config_filter)
    x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val = make_training_data(
        x, cond, config_dataset.train_split, n_batches, BATCH_SIZE)

    # Init model
    config_model = make_config_model(x, hpos)
    params, encoder, decoder, model, h2mu, h2logvar, reparam = init_model(
        rng_model, x, cond, config_model)

    return (
        rng, rng_model, rng_dataset,
        config_norm_x, config_norm_y, config_filter, config_optimisation, config_dataset, config_training, config_model,
        data, x_cols, df,
        x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val,
        total_ds, n_batches, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing,
        params, encoder, decoder, model, h2mu, h2logvar, reparam
    )


def make_loss(loss_type: str, use_l2_reg, use_kl_div, kl_weight):

    if loss_type == 'mse':
        loss_fn = partial(
            loss_wrapper, loss_f=mse_loss, use_l2_reg=use_l2_reg, use_kl_div=use_kl_div, kl_weight=kl_weight)
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented')
    compute_accuracy = partial(accuracy_regression, threshold=0.1)
    return loss_fn, compute_accuracy
