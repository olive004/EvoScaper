import gc
import logging
from typing import Callable, List, Dict, Optional
from copy import deepcopy
from bioreaction.model.data_containers import BasicModel
from bioreaction.model.data_containers import QuantifiedReactions
from bioreaction.misc.misc import load_json_as_dict
import jax
import jaxlib
import jaxlib.xla_extension
import numpy as np
from functools import partial
from datetime import datetime
import pandas as pd
import os

from sklearn.metrics import r2_score
from synbio_morpher.utils.data.data_format_tools.common import write_json
from evoscaper.model.evaluation import (calculate_kl_divergence_aves, estimate_mutual_information_knn, conditional_latent_entropy,
                                        latent_cluster_separation, calculate_distributional_overlap, within_condition_variance_ratio, nearest_neighbor_condition_accuracy)
from evoscaper.model.sampling import sample_reconstructions
from evoscaper.model.vae import sample_z
from evoscaper.scripts.init_from_hpos import init_from_hpos, make_loss, init_model
from evoscaper.scripts.verify import verify
from evoscaper.utils.dataclasses import DatasetConfig, FilterSettings, ModelConfig, NormalizationSettings, OptimizationConfig, TrainingConfig
from evoscaper.utils.dataset import prep_data, concat_conds, load_by_fn, init_data, make_training_data
from evoscaper.utils.math import make_batch_symmetrical_matrices, arrayise
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.optimiser import make_optimiser
from evoscaper.utils.preprocess import make_datetime_str, make_xcols
from evoscaper.utils.simulation import make_rates, prep_sim, sim_core, prep_cfg, update_species_simulated_rates, setup_model_brn, compute_analytics
from evoscaper.utils.train import train
from evoscaper.utils.tuning import make_configs_initial, make_config_model
from evoscaper.utils.visualise import vis_parity, vis_recon_distribution, vis_training


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
                                           optimiser, optimiser_state, config_training,
                                           epochs=config_training.epochs, loss_fn=loss_fn, compute_accuracy=compute_accuracy,
                                           save_every=config_training.print_every, include_params_in_all_saves=False,
                                           patience=config_training.patience, threshold_early_val_acc=config_training.threshold_early_val_acc)

    print('Training complete:', datetime.now() - tstart)

    pred_y = model(params, rng, x_train, cond=cond_train)
    r2_train = r2_score(y_train.flatten(), pred_y.flatten())

    save_path = make_savepath(task=task, top_dir=top_write_dir)
    write_json(deepcopy(saves), out_path=save_path)

    return params, saves, save_path, r2_train, info_early_stop


def vis(saves: dict, x, pred_y, top_write_dir):
    vis_training(saves, save_path=os.path.join(
        top_write_dir, 'training.png'))
    vis_parity(x, pred_y, save_path=os.path.join(
        top_write_dir, 'parity.png'))
    vis_recon_distribution(x, pred_y, len(pred_y), save_path=os.path.join(
        top_write_dir, 'recon_distribution.png'))


def test_conditionality(params, rng, decoder,
                        config_dataset: DatasetConfig, config_norm_x: NormalizationSettings,
                        config_norm_y: NormalizationSettings,
                        config_model,
                        x_datanormaliser, x_methods_preprocessing, cond, n_to_sample=int(1e4)):
    n_categories = config_norm_y.categorical_n_bins
    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=n_categories, n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=config_norm_y.categorical, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max())

    mi = np.mean(jax.vmap(partial(estimate_mutual_information_knn, k=5))(
        z, sampled_cond))
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

    kls = {}
    kde_overlaps = {}
    for idx_obj, obj_col in enumerate(config_dataset.objective_col):
        kl_divs = calculate_kl_divergence_aves(fake_circuits.reshape(
            -1, fake_circuits.shape[-1]), sampled_cond.reshape(-1, sampled_cond.shape[-1])[..., idx_obj])
        kls[obj_col] = np.nanmean(kl_divs)

        if config_norm_x.min_max:
            fake_circuits = np.where(fake_circuits > 1, 1, fake_circuits)
            fake_circuits = np.where(fake_circuits < 0, 0, fake_circuits)

        overlaps = np.zeros(
            (fake_circuits.shape[-1], sampled_cond.shape[0], sampled_cond.shape[0]))
        for ix in range(fake_circuits.shape[-1]):
            kde = calculate_distributional_overlap(fake_circuits.reshape(sampled_cond.shape[0], -1, fake_circuits.shape[-1])[..., ix],
                                                   dist_type='binned')
            overlaps[ix] = kde
        overlaps = np.min(overlaps, axis=0)
        overlaps_nodiag = overlaps[~np.eye(overlaps.shape[0], dtype=bool)
                                   ].reshape(overlaps.shape[0], -1)
        kde_overlaps['Overlap ' + obj_col] = {'mean': overlaps_nodiag.mean(axis=1),
                                              'std': overlaps_nodiag.std(axis=1)}

    return mi, kls, kde_overlaps


def collect_latent_stats(params, rng, encoder, decoder, h2mu, h2logvar, cond, objective_cols, config_dataset, config_model, x_datanormaliser, x_methods_preprocessing, config_norm_y, n_categories, n_to_sample):

    x, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                n_categories=n_categories, n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                objective_cols=config_dataset.objective_col,
                                                use_binned_sampling=config_norm_y.categorical, use_onehot=config_norm_y.categorical_onehot,
                                                cond_min=cond.min(), cond_max=cond.max())

    x = x.reshape(np.prod(x.shape[:-1]), x.shape[-1])
    sampled_cond = sampled_cond.reshape(
        np.prod(sampled_cond.shape[:-1]), sampled_cond.shape[-1])
    h = encoder(params, rng, np.concatenate([x, sampled_cond], axis=-1))
    mu = h2mu(params, rng, h)
    logvar = h2logvar(params, rng, h)
    z = sample_z(mu, logvar, rng, deterministic=False)

    # entropy_val, per_cond_entropy = conditional_latent_entropy(
    #     z, sampled_cond)
    # cluster_sep = latent_cluster_separation(z, sampled_cond)
    # mi_val, mi_per_dim = mutual_information_latent_condition(
    #     z, sampled_cond)
    # variance_ratio = within_condition_variance_ratio(z, sampled_cond)
    # nn_accuracy = nearest_neighbor_condition_accuracy(
    #     z, sampled_cond)

    # latent_stats = {
    #     "conditional_entropy": entropy_val,
    #     "per_condition_entropy": per_cond_entropy,
    #     "cluster_separation": cluster_sep,
    #     "mutual_information": mi_val,
    #     "mutual_information_per_dim": mi_per_dim,
    #     "variance_ratio": variance_ratio,
    #     "nn_accuracy": nn_accuracy
    # }

    latent_stats = {}
    for idx_obj, obj_col in enumerate(objective_cols):
        entropy_val, entropy_std = conditional_latent_entropy(
            z, sampled_cond[..., idx_obj])
        cluster_sep = latent_cluster_separation(z, sampled_cond[..., idx_obj])
        # mi_val, mi_per_dim = mutual_information_latent_condition(
        #     z, sampled_cond[..., idx_obj])
        variance_ratio = within_condition_variance_ratio(
            z, sampled_cond[..., idx_obj])
        nn_accuracy = nearest_neighbor_condition_accuracy(
            z, sampled_cond[..., idx_obj])

        latent_stats.update({
            f"{obj_col}_conditional_entropy": entropy_val,
            f"{obj_col}_condition_entropy_std": entropy_std,
            f"{obj_col}_cluster_separation": cluster_sep,
            # "mutual_information": mi_val,
            # "mutual_information_per_dim": mi_per_dim,
            f"{obj_col}_variance_ratio": variance_ratio,
            f"{obj_col}_nn_accuracy": nn_accuracy
        })
    return latent_stats


def test(model, params, rng, encoder, h2mu, h2logvar, decoder, saves: dict, data_test,
         config_dataset: DatasetConfig, config_norm_x: NormalizationSettings, config_norm_y: NormalizationSettings, config_model: ModelConfig,
         x_cols, config_filter: FilterSettings, top_write_dir,
         x_datanormaliser: DataNormalizer, x_methods_preprocessing,
         y_datanormaliser: DataNormalizer, y_methods_preprocessing, visualise=True):

    df = prep_data(data_test, config_dataset.output_species,
                   config_dataset.objective_col, x_cols, config_filter)
    x = np.array(x_datanormaliser.create_chain_preprocessor(x_methods_preprocessing)(
        np.concatenate([df[i].values[:, None] for i in x_cols], axis=1).squeeze()))

    cond = concat_conds(config_dataset.objective_col, df,
                        y_datanormaliser, y_methods_preprocessing)

    pred_y = model(params, rng, x, cond)

    r2_test = r2_score(x.flatten(), pred_y.flatten())

    if visualise:
        vis(saves, x, pred_y, top_write_dir)

    try:
        mi, kl_div_ave, kde_overlap = test_conditionality(params, rng, decoder,
                                                          config_dataset, config_norm_x, config_norm_y, config_model,
                                                          x_datanormaliser, x_methods_preprocessing, cond)
    except Exception as e:
        print(f'Error in test_conditionality: {e}')
        mi, kl_div_ave, kde_overlap = None, None, None

    try:
        latent_stats = collect_latent_stats(
            params, rng, encoder, decoder, h2mu, h2logvar, cond, config_dataset.objective_col,
            config_dataset, config_model, x_datanormaliser, x_methods_preprocessing, config_norm_y,
            n_categories=10, n_to_sample=int(1e4))
    except ValueError as e:
        print(f'Error in collect_latent_stats: {e}')
        latent_stats = {}
    return r2_test, mi, kl_div_ave, kde_overlap, latent_stats


def save_stats(hpos: pd.Series, save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, kl_div_ave: float, kde_overlap: Optional[dict], latent_stats: Dict[str, float], n_layers_enc, n_layers_dec, info_early_stop):
    for k, v in zip(
        ['filename_saved_model', 'total_ds', 'n_batches', 'R2_train', 'R2_test',
            'mutual_information_conditionality', 'kl_div_ave', 'n_layers_enc', 'n_layers_dec', 'info_early_stop'],
            [save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, kl_div_ave, n_layers_enc, n_layers_dec, info_early_stop]):
        hpos[k] = v

    for k, v in latent_stats.items():
        hpos[k] = v

    if kde_overlap is not None:
        for k, v in kde_overlap.items():
            hpos[k] = v
    return hpos


def cvae_scan_single(hpos: pd.Series, top_write_dir=TOP_WRITE_DIR, skip_verify=False, debug=False,
                     visualise=True):

    (
        rng, rng_model, rng_dataset,
        config_norm_x, config_norm_y, config_filter, config_optimisation, config_dataset, config_training, config_model,
        data, x_cols, df,
        x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val,
        total_ds, n_batches, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing, y_datanormaliser, y_methods_preprocessing,
        params, encoder, decoder, model, h2mu, h2logvar, reparam
    ) = init_from_hpos(hpos)

    if not os.path.exists(top_write_dir):
        os.makedirs(top_write_dir, exist_ok=True)

    # Losses
    loss_fn, compute_accuracy = make_loss(
        config_training.loss_func, config_training.use_l2_reg, config_training.use_kl_div, config_training.kl_weight)

    # Train
    params, saves, save_path, r2_train, info_early_stop = train_full(params, rng, model, x_train, cond_train, y_train, x_val, cond_val, y_val,
                                                                     config_optimisation, config_training, loss_fn, compute_accuracy, n_batches,
                                                                     task=f'_hpo_{hpos.loc["index"]}', top_write_dir=top_write_dir)

    # Test & Visualise
    if config_dataset.use_test_data:
        data_test = pd.read_csv(config_dataset.filenames_verify_table) if config_dataset.filenames_verify_table.endswith(
            '.csv') else pd.read_json(config_dataset.filenames_verify_table)
    else:
        print(
            f'Warning: not using the test data for evaluation, but the training data instead of {config_dataset.filenames_verify_table}')
        data_test = data

    r2_test, mi, kl_div_ave, kde_overlap, latent_stats = test(model, params, rng, encoder, h2mu, h2logvar, decoder, saves, data_test,
                                                              config_dataset, config_norm_x, config_norm_y, config_model,
                                                              x_cols, config_filter, top_write_dir,
                                                              x_datanormaliser, x_methods_preprocessing,
                                                              y_datanormaliser, y_methods_preprocessing, visualise=(not (debug) or visualise))

    # Save stats
    hpos = save_stats(hpos, save_path, total_ds, n_batches, r2_train, r2_test, mi, kl_div_ave, kde_overlap, latent_stats,
                      len(config_model.enc_layers), len(config_model.dec_layers), info_early_stop)

    # Verification
    # if True:
    if not skip_verify:
        if (r2_test > 0.8) or (r2_train > 0.8):
            val_config = load_json_as_dict(
                config_dataset.filenames_train_config)
            config_bio = {}
            for k in [kk for kk in val_config['base_configs_ensemble'].keys() if 'vis' not in kk]:
                config_bio.update(val_config['base_configs_ensemble'][k])
            verify(params, rng, decoder,
                   cond,
                   config_bio,
                   config_norm_y,
                   config_dataset,
                   config_model,
                   x_datanormaliser, x_methods_preprocessing,
                   output_species=config_dataset.output_species,
                   signal_species=config_dataset.signal_species,
                   input_species=data[data['sample_name'].notna()
                                      ]['sample_name'].unique(),
                   n_to_sample=int(hpos.loc['eval_n_to_sample']),
                   visualise=True,
                   top_write_dir=top_write_dir)
    return hpos


def loop_scans(df_hpos: pd.DataFrame, top_dir: str, skip_verify=False, debug=False, visualise=True):
    os.makedirs(top_dir, exist_ok=True)
    for i in range(len(df_hpos)):
        hpos = df_hpos.reset_index().iloc[i]
        top_write_dir = os.path.join(
            top_dir, 'cvae_scan', f'hpo_{hpos["index"]}')
        os.makedirs(top_write_dir, exist_ok=True)
        if (hpos.loc['run_successful'] != 'TO_BE_RECORDED') or (hpos.loc['R2_train'] != 'TO_BE_RECORDED'):
            if os.path.exists(hpos['filename_saved_model']):
                dest_path = os.path.join(
                    top_write_dir, os.path.basename(hpos['filename_saved_model']))
                os.system(f"cp {hpos['filename_saved_model']} {dest_path}")
                hpos['filename_saved_model'] = dest_path
        else:
            # hpos['use_grad_clipping'] = True
            if debug:
                hpos = cvae_scan_single(
                    hpos, top_write_dir=top_write_dir, skip_verify=skip_verify,
                    visualise=visualise)
            else:
                try:
                    hpos = cvae_scan_single(
                        hpos, top_write_dir=top_write_dir, skip_verify=skip_verify,
                        visualise=visualise)
                    hpos.loc['run_successful'] = True
                    hpos.loc['error_msg'] = ''
                except Exception as e:
                    print("Try 1", e)
                    if ('nan' in str(e).lower()) and (hpos.loc['use_grad_clipping'] == False):
                        try:
                            hpos['use_grad_clipping'] = True
                            hpos = cvae_scan_single(
                                hpos, top_write_dir=top_write_dir,
                                visualise=visualise)
                            hpos.loc['run_successful'] = True
                            hpos.loc['error_msg'] = ''
                        except Exception as e:
                            print("Try 2", e)
                            hpos.loc['run_successful'] = False
                            hpos.loc['error_msg'] = str(e)
                    else:
                        hpos.loc['run_successful'] = False
                        hpos.loc['error_msg'] = str(e)
                except:
                    hpos.loc['run_successful'] = False
                    hpos.loc['error_msg'] = 'sys exit'

        hpos = pd.Series(hpos) if type(
            hpos) == dict else hpos.drop('index')
        for c in hpos.index:
            if c not in df_hpos.columns:
                df_hpos.loc[:, c] = 'TO_BE_RECORDED'
        
        df_hpos.iloc[i] = hpos
        df_hpos.to_csv(os.path.join(top_dir, 'df_hpos.csv'))
        write_json(df_hpos.to_dict(), os.path.join(
            top_dir, 'df_hpos.json'), overwrite=True)
        df_hpos.to_json(os.path.join(top_dir, 'df_hpos2.json'), lines=True)
        gc.collect()
    return df_hpos


def get_input_species(data):
    return data[data['sample_name'].notna()]['sample_name'].unique()


def load_params(fn_saves):
    saves_loaded = load_json_as_dict(fn_saves)
    params = saves_loaded[str(list(saves_loaded.keys())[-1])]['params']
    return arrayise(params)


def sample_models(hpos, datasets):
    """Run hyperparameter optimization for a single set of parameters"""
    data = datasets[hpos['filenames_train_table']]

    rng = jax.random.PRNGKey(int(hpos['seed_train']))
    rng_model = jax.random.PRNGKey(int(hpos['seed_arch']))
    rng_dataset = jax.random.PRNGKey(int(hpos['seed_dataset']))

    # Configs + data
    (config_norm_x, config_norm_y, config_filter, config_optimisation,
     config_dataset, config_training) = make_configs_initial(hpos)

    x_cols = make_xcols(data, config_dataset.x_type,
                        config_dataset.include_diffs)

    # Init data
    (df, x, cond, total_ds, n_batches, BATCH_SIZE, x_datanormaliser, x_methods_preprocessing,
     y_datanormaliser, y_methods_preprocessing) = init_data(
        data, x_cols, config_dataset.objective_col, config_dataset.output_species, config_dataset.total_ds_max,
        config_training.batch_size, rng_dataset, config_norm_x, config_norm_y, config_filter)
    x, cond, y, x_train, cond_train, y_train, x_val, cond_val, y_val = make_training_data(
        x, cond, config_dataset.train_split, n_batches, BATCH_SIZE)

    # Init model
    config_model = make_config_model(x, hpos)

    params_init, encoder, decoder, model, h2mu, h2logvar, reparam = init_model(
        rng_model, x, cond, config_model)

    try:
        params = load_params(hpos['filename_saved_model'])
    except:
        params = load_params(os.path.join(
            'notebooks', hpos['filename_saved_model']))

    # Generate fake circuits
    n_categories = config_norm_y.categorical_n_bins if 'eval_n_categories' not in hpos.index else hpos[
        'eval_n_categories']
    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=n_categories if n_categories is not None else 10,
                                                            n_to_sample=int(
                                                                hpos['eval_n_to_sample']),
                                                            hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser,
                                                            x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=config_norm_y.categorical,
                                                            use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=hpos['eval_cond_min'],
                                                            cond_max=hpos['eval_cond_max'],
                                                            clip_range=None)

    return fake_circuits, z, sampled_cond


def run_sim_multi(fake_circuits_reshaped: np.ndarray, forward_rates: np.ndarray, reverse_rates: np.ndarray, signal_species: List[str],
                  config_bio: dict, model_brn: BasicModel, qreactions: QuantifiedReactions, ordered_species: list, results_dir: str):

    # Process circuits and simulate
    model_brn, qreactions = update_species_simulated_rates(
        ordered_species, forward_rates[0], reverse_rates[0], model_brn, qreactions)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time, threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        signal_species, qreactions, fake_circuits_reshaped,
        config_bio, forward_rates, reverse_rates)

    ys, ts, y0m, y00s, ts0 = sim_core(y00, forward_rates[0], reverse_rates, qreactions, signal_onehot, signal_target,
                                      t0, t1, dt0, dt1, save_steps, max_steps, stepsize_controller, threshold=threshold_steady_states,
                                      total_time=total_time, disable_logging=False)
    for k, v in zip(['ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy'], [ys, ts, y0m, y00s, ts0]):
        np.save(os.path.join(results_dir, k), v)

    try:
        analytics = jax.vmap(partial(compute_analytics, t=ts, labels=np.arange(
            ys.shape[-1]), signal_onehot=signal_onehot))(ys)
    except jaxlib.xla_extension.XlaRuntimeError:
        logging.warning(
            'Could not compute analytics due to resource constraints.')
        analytics = {}

    return analytics, ys, ts, y0m, y00s, ts0


def save(results_dir, analytics, ys, ts, y0m, y00s, ts0):

    write_json(analytics, os.path.join(results_dir, 'analytics.json'))
    for k, v in zip(['ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy'], [ys, ts, y0m, y00s, ts0]):
        np.save(os.path.join(results_dir, k), v)


# Run simulation for each successful HPO
def generate_all_fake_circuits(df_hpos, datasets, input_species, postprocs: dict):
    successful_runs = df_hpos[(df_hpos['run_successful'] == True) | (
        df_hpos['R2_train'].apply(lambda x: x > 0.8 if type(x) == float else False))]

    if len(successful_runs) == 0:
        raise ValueError(f'No successful runs from ML scan: {df_hpos}')

    n_runs = len(successful_runs)

    fake_circuits_list = [None] * n_runs
    z_list = [None] * n_runs
    sampled_cond_list = [None] * n_runs
    forward_rates_list = [None] * n_runs
    reverse_rates_list = [None] * n_runs

    # Generate circuits from each model
    for idx, (_, hpos) in enumerate(successful_runs.iterrows()):
        fake_circuits, z, sampled_cond = sample_models(hpos, datasets)

        fake_circuits_reshaped = make_batch_symmetrical_matrices(
            fake_circuits.reshape(-1, fake_circuits.shape[-1]),
            side_length=len(input_species))

        forward_rates, reverse_rates = make_rates(
            hpos['x_type'], fake_circuits_reshaped, postprocs)

        fake_circuits_list[idx] = fake_circuits_reshaped
        # z_list[idx] = z
        sampled_cond_list[idx] = sampled_cond
        forward_rates_list[idx] = forward_rates
        reverse_rates_list[idx] = reverse_rates

    # Mixing all circuits together
    all_fake_circuits = np.concatenate(fake_circuits_list, axis=0)
    all_forward_rates = np.concatenate(forward_rates_list, axis=0)
    all_reverse_rates = np.concatenate(reverse_rates_list, axis=0)
    # all_z = dict(zip(successful_runs.index.tolist(), z_list))
    all_sampled_cond = dict(
        zip(successful_runs.index.tolist(), sampled_cond_list))
    # all_sampled_cond = np.concatenate(sampled_cond_list, axis=0)

    del fake_circuits_list, z_list, sampled_cond_list, forward_rates_list, reverse_rates_list

    return all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond  # , all_z


def extend_analytics(analytics: dict, analytics_i: dict):
    for k in analytics_i.keys():
        if k not in analytics.keys():
            analytics[k] = analytics_i[k][None, :]
        else:
            analytics[k] = np.concatenate(
                [analytics[k], analytics_i[k][None, :]])
    return analytics


def sim_all_models(config_multisim,
                   df_hpos, datasets, input_species,
                   top_write_dir,
                   config_bio):

    model_brn, qreactions, postprocs, ordered_species = setup_model_brn(
        config_bio, input_species)

    batch_size = config_multisim['eval_batch_size']
    all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond = generate_all_fake_circuits(
        df_hpos, datasets, input_species, postprocs)
    np.save(os.path.join(top_write_dir, 'fake_circuits.npy'), all_fake_circuits)
    # np.save(os.path.join(top_write_dir, 'sampled_cond.npy'), all_sampled_cond)
    os.makedirs(os.path.join(top_write_dir, 'sampled_cond'), exist_ok=True)
    for i in all_sampled_cond.keys():
        np.save(os.path.join(top_write_dir, 'sampled_cond',
                f'sampled_cond_{i}'), all_sampled_cond[i])

    n_batches = int(np.ceil(len(all_fake_circuits) / batch_size))
    time_start = datetime.now()
    batch_dir = os.path.join(top_write_dir, 'batch_results')
    os.makedirs(batch_dir, exist_ok=True)
    analytics, ys, ts, y0m, y00s, ts0 = {}, None, None, None, None, None
    for i in range(n_batches):
        results_dir = os.path.join(batch_dir, f'batch_{i}')
        os.makedirs(results_dir, exist_ok=True)

        i1, i2 = i*batch_size, (i+1)*batch_size
        logging.info(
            f'Simulating batch {i+1} of {n_batches} ({datetime.now() - time_start})')
        time_sim = datetime.now()
        analytics, ys, ts, y0m, y00s, ts0 = run_sim_multi(
            all_fake_circuits[i1:i2], all_forward_rates[i1:i2],
            all_reverse_rates[i1:i2], df_hpos['signal_species'].iloc[0],
            config_bio, model_brn, qreactions, ordered_species,
            results_dir)
        logging.info(
            f'Simulation complete for batch {i+1} of {n_batches} (took {datetime.now() - time_sim})')

        # Save results
        save(results_dir, analytics, ys, ts, y0m, y00s, ts0)
    return analytics, ys, ts, y0m, y00s, ts0, all_fake_circuits, all_sampled_cond


def cvae_scan_multi(df_hpos: pd.DataFrame, fn_config_multisim: str, top_write_dir=TOP_WRITE_DIR, debug=False, visualise=True):
    """Run multiple CVAE scans and evaluate generated circuits"""
    os.makedirs(top_write_dir, exist_ok=True)

    # First run all models and save results
    df_hpos = loop_scans(df_hpos, top_write_dir,
                         skip_verify=True, debug=debug, visualise=visualise)

    # Global options
    config_multisim = load_json_as_dict(fn_config_multisim)
    config_bio = load_json_as_dict(
        config_multisim['filename_simulation_settings'])
    # config_bio = {}
    # if 'base_configs_ensemble' in val_config.keys():
    #     for k in [kk for kk in val_config['base_configs_ensemble'].keys() if 'vis' not in kk]:
    #         config_bio.update(val_config['base_configs_ensemble'][k])
    # else:
    #     config_bio = val_config

    # Pre-load datasets
    datasets = {k: load_by_fn(k)
                for k in df_hpos['filenames_train_table'].unique()}
    input_species = get_input_species(
        datasets[df_hpos['filenames_train_table'].unique()[0]])

    config_bio = prep_cfg(config_bio, input_species)

    # Simulate successful runs all in one go
    analytics, ys, ts, y0m, y00s, ts0, all_fake_circuits, all_sampled_cond = sim_all_models(
        config_multisim,
        df_hpos, datasets, input_species,
        top_write_dir,
        config_bio)

    return df_hpos, analytics, ys, ts, y0m, y00s, ts0, all_fake_circuits, all_sampled_cond
