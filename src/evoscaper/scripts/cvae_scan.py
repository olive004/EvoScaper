

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from datetime import datetime
import pandas as pd
import os

from synbio_morpher.utils.data.data_format_tools.common import write_json
from evoscaper.model.vae import sample_z, VAE_fn
from evoscaper.model.shared import arrayise, get_activation_fn
from evoscaper.model.loss import loss_wrapper, compute_accuracy_regression, mse_loss, accuracy_regression
from evoscaper.utils.dataclasses import DatasetConfig, NormalizationSettings, ModelConfig
from evoscaper.utils.dataset import init_data
from evoscaper.utils.math import convert_to_scientific_exponent
from evoscaper.utils.optimiser import make_optimiser
from evoscaper.utils.preprocess import make_xcols
from evoscaper.utils.train import train
from evoscaper.utils.tuning import make_configs_initial, make_config_model


def init_model(rng, x, cond, config_model: ModelConfig):
    # rng = jax.random.PRNGKey(config_model.seed)

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


def load_data(config_dataset: DatasetConfig):
    data = pd.read_csv(config_dataset.filenames_train_table)
    X_COLS = make_xcols(data, config_dataset.x_type,
                        config_dataset.include_diffs)
    return data, X_COLS


def make_savepath(task = '_test'):
    save_path = str(datetime.now()).split(' ')[0].replace(
        '-', '_') + '__' + str(datetime.now()).split(' ')[-1].split('.')[0].replace(':', '_') + '_saves' + task
    save_path = os.path.join('weight_saves', '01_cvae', save_path)
    return save_path


def make_loss(loss_type: str, use_l2_reg, use_kl_div, kl_weight):

    if loss_type == 'mse':
        loss_fn = partial(
            loss_wrapper, loss_f=mse_loss, use_l2_reg=use_l2_reg, use_kl_div=use_kl_div, kl_weight=kl_weight)
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented')
    compute_accuracy = partial(accuracy_regression, rtol=1e-3, atol=1e-5)
    return loss_fn, compute_accuracy


def main(hpos):

    rng = jax.random.PRNGKey(hpos['seed'])
    rng_model = jax.random.PRNGKey(hpos['seed_arch'])

    # Configs + data
    (config_norm_x, config_norm_y, config_filter, config_dataset,
     config_training) = make_configs_initial(hpos)
    data, X_COLS = load_data(config_dataset)

    # Init data
    (df, x, cond, TOTAL_DS, N_BATCHES, x_datanormaliser, x_methods_preprocessing,
     y_datanormaliser, y_methods_preprocessing) = init_data(
         data, X_COLS, config_dataset.objective_col, config_dataset.output_species, config_dataset.total_ds_max,
         config_training.batch_size, config_training.seed_dataset, config_norm_x, config_norm_y, config_filter)

    # Init model
    config_model = make_config_model(x, hpos)

    params, encoder, decoder, model, h2mu, h2logvar, reparam = init_model(
        rng_model, x, cond, config_model)
    x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val = make_training_data(
        x, cond, y, config_dataset.train_split, N_BATCHES, config_training.batch_size)

    optimiser, optimiser_state = init_optimiser(x, config_training.learning_rate_sched, config_training.learning_rate,
                                                config_training.epochs, config_training.l2_reg_alpha, config_training.use_warmup,
                                                config_training.warmup_epochs, N_BATCHES)
    
    # Losses
    loss_fn, compute_accuracy = make_loss(
        config_training.loss_func, config_training.use_l2_reg, config_training.use_kl_div, config_training.kl_weight)
    
    # Train
    tstart = datetime.now()
    params, saves = train(params, rng, model,
                          x_train, cond_train, y_train, x_val, cond_val, y_val,
                          optimiser, optimiser_state,
                          use_l2_reg=config_training.use_l2_reg, l2_reg_alpha=config_training.l2_reg_alpha,
                          epochs=config_training.epochs, loss_fn=loss_fn, compute_accuracy=compute_accuracy,
                          save_every=config_training.print_every, include_params_in_saves=False)
    print('Training complete:', datetime.now() - tstart)
    
    save_path = make_savepath()
    write_json(saves, out_path=save_path)
    print(save_path)
    
    # Verification
    
