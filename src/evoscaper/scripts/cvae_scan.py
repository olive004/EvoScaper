

from copy import deepcopy
from typing import Callable
from bioreaction.misc.misc import load_json_as_dict
import haiku as hk
import jax
import numpy as np
from functools import partial
from datetime import datetime
import pandas as pd
import os

from sklearn.metrics import r2_score
from synbio_morpher.utils.data.data_format_tools.common import write_json
from evoscaper.model.evaluation import estimate_mutual_information_knn
from evoscaper.model.sampling import sample_reconstructions
from evoscaper.model.vae import VAE_fn
from evoscaper.model.shared import get_activation_fn
from evoscaper.scripts.init_from_hpos import init_from_hpos, make_loss
from evoscaper.scripts.verify import verify
from evoscaper.utils.dataclasses import DatasetConfig, FilterSettings, ModelConfig, NormalizationSettings, OptimizationConfig, TrainingConfig
from evoscaper.utils.dataset import prep_data, concat_conds
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.optimiser import make_optimiser
from evoscaper.utils.preprocess import make_datetime_str, make_xcols
from evoscaper.utils.train import train
from evoscaper.utils.visualise import vis_histplot_combined_realfake, vis_parity, vis_recon_distribution, vis_training


TOP_WRITE_DIR = 'data'


def make_savepath(task='_test', top_dir=TOP_WRITE_DIR):
    save_path = make_datetime_str() + '_saves' + task
    save_path = os.path.join(top_dir, save_path)
    return save_path


def init_optimiser(params, learning_rate_sched, learning_rate, epochs, l2_reg_alpha, use_warmup, warmup_epochs, n_batches, opt_method):
    optimiser = make_optimiser(learning_rate_sched, learning_rate,
                               epochs, l2_reg_alpha, use_warmup, warmup_epochs, n_batches,
                               opt_method)
    optimiser_state = optimiser.init(params)
    return optimiser, optimiser_state


def train_full(params, rng, model,
               x_train, cond_train, y_train, x_val, cond_val, y_val,
               config_optimisation: OptimizationConfig, config_training: TrainingConfig,
               loss_fn: Callable, compute_accuracy: Callable, n_batches: int, task: str, top_write_dir: str):
    optimiser, optimiser_state = init_optimiser(params, config_optimisation.learning_rate_sched, config_training.learning_rate,
                                                config_training.epochs, config_training.l2_reg_alpha, config_optimisation.use_warmup,
                                                config_optimisation.warmup_epochs, n_batches, config_optimisation.opt_method)
    tstart = datetime.now()
    params, saves, info_early_stop = train(params, rng, model,
                                           x_train, cond_train, y_train, x_val, cond_val, y_val,
                                           optimiser, optimiser_state,
                                           use_l2_reg=config_training.use_l2_reg, l2_reg_alpha=config_training.l2_reg_alpha,
                                           epochs=config_training.epochs, loss_fn=loss_fn, compute_accuracy=compute_accuracy,
                                           save_every=config_training.print_every, include_params_in_all_saves=False)

    print('Training complete:', datetime.now() - tstart)

    pred_y = model(params, rng, x_train, cond=cond_train)
    r2_train = r2_score(y_train.flatten(), pred_y.flatten())

    save_path = make_savepath(task=task, top_dir=top_write_dir)
    write_json(deepcopy(saves), out_path=save_path)

    return params, saves, save_path, r2_train, info_early_stop


def vis(saves, x, pred_y, top_write_dir):
    vis_training(saves, save_path=os.path.join(
        top_write_dir, 'training.png'))
    vis_parity(x, pred_y, save_path=os.path.join(
        top_write_dir, 'parity.png'))
    vis_recon_distribution(x, pred_y, len(pred_y), save_path=os.path.join(
        top_write_dir, 'recon_distribution.png'))


def test_conditionality(params, rng, decoder, df, x_cols,
                        config_dataset: DatasetConfig, config_norm_y: NormalizationSettings, config_model, top_write_dir,
                        x_datanormaliser, x_methods_preprocessing,
                        y_datanormaliser, y_methods_preprocessing, cond):
    n_categories = config_norm_y.categorical_n_bins
    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=n_categories, n_to_sample=10000, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=config_norm_y.categorical, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max())

    mi = jax.vmap(partial(estimate_mutual_information_knn, k=5))(
        z, sampled_cond)
    # mi = estimate_mutual_information_knn(z.reshape(np.prod(
    #     z.shape[:-1]), z.shape[-1]), sampled_cond.reshape(np.prod(sampled_cond.shape[:-1]), sampled_cond.shape[-1]), k=5)

    # vis_histplot_combined_realfake(n_categories, df, x_cols, config_dataset.objective_col,
    #                                y_datanormaliser, y_methods_preprocessing,
    #                                fake_circuits, z, sampled_cond, config_norm_y.categorical_onehot,
    #                                save_path=os.path.join(top_write_dir, 'combined_fill.png'), multiple='fill', fill=True)
    # vis_histplot_combined_realfake(n_categories, df, x_cols, config_dataset.objective_col,
    #                                y_datanormaliser, y_methods_preprocessing,
    #                                fake_circuits, z, sampled_cond, config_norm_y.categorical_onehot,
    #                                save_path=os.path.join(top_write_dir, 'combined_layer.png'), multiple='layer', fill=False)
    return mi


def test(model, params, rng, decoder, saves, data_test,
         config_dataset: DatasetConfig, config_norm_y: NormalizationSettings, config_model: ModelConfig,
         x_cols, config_filter: FilterSettings, top_write_dir,
         x_datanormaliser: DataNormalizer, x_methods_preprocessing,
         y_datanormaliser: DataNormalizer, y_methods_preprocessing):

    df = prep_data(data_test, config_dataset.output_species,
                   config_dataset.objective_col, x_cols, config_filter)
    x = x_datanormaliser.create_chain_preprocessor(x_methods_preprocessing)(
        np.concatenate([df[i].values[:, None] for i in x_cols], axis=1).squeeze())
    
    cond = concat_conds(config_dataset.objective_col, df, y_datanormaliser, y_methods_preprocessing)

    pred_y = model(params, rng, x, cond)

    r2_test = r2_score(x.flatten(), pred_y.flatten())

    vis(saves, x, pred_y, top_write_dir)

    mi = test_conditionality(params, rng, decoder, df, x_cols,
                             config_dataset, config_norm_y, config_model, top_write_dir,
                             x_datanormaliser, x_methods_preprocessing,
                             y_datanormaliser, y_methods_preprocessing, cond)

    return r2_test, mi


def save_stats(hpos: pd.Series, save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, n_layers_enc, n_layers_dec, info_early_stop):
    for k, v in zip(
        ['filename_saved_model', 'total_ds', 'n_batches', 'R2_train', 'R2_test',
            'mutual_information_conditionality', 'n_layers_enc', 'n_layers_dec', 'info_early_stop'],
            [save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, n_layers_enc, n_layers_dec, info_early_stop]):
        hpos[k] = v
    return hpos


def main(hpos: pd.Series, top_write_dir=TOP_WRITE_DIR):

    (
        rng, rng_model, rng_dataset,
        config_norm_x, config_norm_y, config_filter, config_optimisation, config_dataset, config_training, config_model,
        data, x_cols, df,
        x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val,
        total_ds, n_batches, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing,
        params, encoder, decoder, model, h2mu, h2logvar, reparam
    ) = init_from_hpos(hpos)

    if not os.path.exists(top_write_dir):
        os.makedirs(top_write_dir)

    # Losses
    loss_fn, compute_accuracy = make_loss(
        config_training.loss_func, config_training.use_l2_reg, config_training.use_kl_div, config_training.kl_weight)

    # Train
    params, saves, save_path, r2_train, info_early_stop = train_full(params, rng, model, x_train, cond_train, y_train, x_val, cond_val, y_val,
                                                                     config_optimisation, config_training, loss_fn, compute_accuracy, n_batches,
                                                                     task=f'_hpo_{hpos["index"]}', top_write_dir=top_write_dir)

    # Test & Visualise
    if config_dataset.use_test_data:
        data_test = pd.read_csv(config_dataset.filenames_verify_table)
    else:
        print(
            f'Warning: not using the test data for evaluation, but the training data instead of {config_dataset.filenames_verify_table}')
        data_test = data

    r2_test, mi = test(model, params, rng, decoder, saves, data_test,
                       config_dataset, config_norm_y, config_model,
                       x_cols, config_filter, top_write_dir,
                       x_datanormaliser, x_methods_preprocessing,
                       y_datanormaliser, y_methods_preprocessing)

    # Save stats
    hpos = save_stats(hpos, save_path, total_ds, n_batches, r2_train, r2_test, mi, len(
        config_model.enc_layers), len(config_model.dec_layers), info_early_stop)

    # Verification
    # if True:  
    if (r2_test > 0.8) or (r2_train > 0.8):
        val_config = load_json_as_dict(config_dataset.filenames_train_config)
        config_bio = {}
        for k in [kk for kk in val_config['base_configs_ensemble'].keys() if 'vis' not in kk]:
            config_bio.update(val_config['base_configs_ensemble'][k])
        verify(params, rng, decoder,
               df, cond,
               config_bio,
               config_norm_y,
               config_dataset,
               config_model,
               x_datanormaliser, x_methods_preprocessing,
               y_datanormaliser,
               output_species=config_dataset.output_species,
               signal_species=config_dataset.signal_species,
               input_species=data[data['sample_name'].notna()
                                  ]['sample_name'].unique(),
               n_to_sample=int(hpos['eval_n_to_sample']),
               visualise=True,
               top_write_dir=top_write_dir)
    return hpos
