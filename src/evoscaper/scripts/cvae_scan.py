import logging
from typing import Callable, List
from copy import deepcopy
from bioreaction.model.data_containers import BasicModel
from bioreaction.model.data_containers import QuantifiedReactions
from bioreaction.misc.misc import load_json_as_dict
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
from evoscaper.scripts.init_from_hpos import init_from_hpos, make_loss, init_model
from evoscaper.scripts.verify import verify
from evoscaper.utils.dataclasses import DatasetConfig, FilterSettings, ModelConfig, NormalizationSettings, OptimizationConfig, TrainingConfig
from evoscaper.utils.dataset import prep_data, concat_conds, load_by_fn, init_data, make_training_data
from evoscaper.utils.math import make_batch_symmetrical_matrices, make_flat_triangle
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.optimiser import make_optimiser
from evoscaper.utils.preprocess import make_datetime_str, make_xcols
from evoscaper.utils.simulation import make_rates, prep_sim, sim, prep_cfg, update_species_simulated_rates, setup_model_brn
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
                        y_datanormaliser, y_methods_preprocessing, cond, n_to_sample=1000):
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
    return mi


def test(model, params, rng, decoder, saves, data_test,
         config_dataset: DatasetConfig, config_norm_y: NormalizationSettings, config_model: ModelConfig,
         x_cols, config_filter: FilterSettings, top_write_dir,
         x_datanormaliser: DataNormalizer, x_methods_preprocessing,
         y_datanormaliser: DataNormalizer, y_methods_preprocessing, visualise=True):

    df = prep_data(data_test, config_dataset.output_species,
                   config_dataset.objective_col, x_cols, config_filter)
    x = x_datanormaliser.create_chain_preprocessor(x_methods_preprocessing)(
        np.concatenate([df[i].values[:, None] for i in x_cols], axis=1).squeeze())

    cond = concat_conds(config_dataset.objective_col, df,
                        y_datanormaliser, y_methods_preprocessing)

    pred_y = model(params, rng, x, cond)

    r2_test = r2_score(x.flatten(), pred_y.flatten())

    if visualise:
        vis(saves, x, pred_y, top_write_dir)

    try:
        mi = test_conditionality(params, rng, decoder, df, x_cols,
                                 config_dataset, config_norm_y, config_model, top_write_dir,
                                 x_datanormaliser, x_methods_preprocessing,
                                 y_datanormaliser, y_methods_preprocessing, cond)
    except Exception as e:
        print(f'Error in test_conditionality: {e}')
        mi = None

    return r2_test, mi


def save_stats(hpos: pd.Series, save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, n_layers_enc, n_layers_dec, info_early_stop):
    for k, v in zip(
        ['filename_saved_model', 'total_ds', 'n_batches', 'R2_train', 'R2_test',
            'mutual_information_conditionality', 'n_layers_enc', 'n_layers_dec', 'info_early_stop'],
            [save_path, total_ds, n_batches, r2_train, r2_test, mutual_information_conditionality, n_layers_enc, n_layers_dec, info_early_stop]):
        hpos[k] = v
    return hpos


def cvae_scan_single(hpos: pd.Series, top_write_dir=TOP_WRITE_DIR, skip_verify=False):

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
        data_test = pd.read_csv(config_dataset.filenames_verify_table) if config_dataset.filenames_verify_table.endswith(
            '.csv') else pd.read_json(config_dataset.filenames_verify_table)
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
    if not skip_verify:
        if (r2_test > 0.8) or (r2_train > 0.8):
            val_config = load_json_as_dict(
                config_dataset.filenames_train_config)
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


def loop_scans(df_hpos: pd.DataFrame, top_dir: str, skip_verify=False, debug=False):
    for i in range(len(df_hpos)):
        hpos = df_hpos.reset_index().iloc[i]
        top_write_dir = os.path.join(
            top_dir, 'cvae_scan', f'hpo_{hpos["index"]}')
        # hpos['use_grad_clipping'] = True
        if debug:
            hpos = cvae_scan_single(
                hpos, top_write_dir=top_write_dir, skip_verify=skip_verify)
        else:
            try:
                hpos = cvae_scan_single(
                    hpos, top_write_dir=top_write_dir, skip_verify=skip_verify)
                hpos.loc['run_successful'] = True
                hpos.loc['error_msg'] = ''
            except Exception as e:
                print("Try 1", e)
                if 'nan' in str(e).lower() and (hpos['use_grad_clipping'] == False):
                    try:
                        hpos['use_grad_clipping'] = True
                        hpos = cvae_scan_single(
                            hpos, top_write_dir=top_write_dir)
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

        df_hpos.iloc[i] = pd.Series(hpos) if type(
            hpos) == dict else hpos.drop('index')
        # df_hpos.loc[i] = pd.DataFrame.from_dict(hpos).drop('index')
        os.makedirs(top_dir, exist_ok=True)
        df_hpos.to_csv(os.path.join(top_dir, 'df_hpos.csv'))
        write_json(df_hpos.to_dict(), os.path.join(
            top_dir, 'df_hpos.json'), overwrite=True)
    return df_hpos


def get_input_species(data):
    return data[data['sample_name'].notna()]['sample_name'].unique()


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
    params, encoder, decoder, model, h2mu, h2logvar, reparam = init_model(
        rng_model, x, cond, config_model)

    # Generate fake circuits
    n_categories = config_norm_y.categorical_n_bins if 'eval_n_categories' not in hpos.index else hpos['eval_n_categories']
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


def run_sim_multi(fake_circuits_reshaped: np.ndarray, forward_rates: np.ndarray, reverse_rates: np.ndarray, signal_species: List[str], config_bio: dict, model_brn: BasicModel, qreactions: QuantifiedReactions, ordered_species: list):

    # Process circuits and simulate
    model_brn, qreactions = update_species_simulated_rates(
        ordered_species, forward_rates[0], reverse_rates[0], model_brn, qreactions)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time, threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        signal_species, qreactions, fake_circuits_reshaped,
        config_bio, forward_rates, reverse_rates)

    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates, qreactions, signal_onehot, signal_target,
                                            t0, t1, dt0, dt1, save_steps, max_steps, stepsize_controller, threshold=threshold_steady_states, total_time=total_time)

    return analytics, ys, ts, y0m, y00s, ts0


def save(results_path, analytics, ys, ts, y0m, y00s, ts0, fake_circuits, sampled_cond):
    write_json(analytics, os.path.join(results_path, 'analytics.json'))
    np.save(os.path.join(results_path, 'ys.npy'), ys)
    np.save(os.path.join(results_path, 'y0m.npy'), y0m)
    np.save(os.path.join(results_path, 'y00s.npy'), y00s)
    np.save(os.path.join(results_path, 'ts.npy'), ts)
    np.save(os.path.join(results_path, 'ts0.npy'), ts0)
    np.save(os.path.join(results_path, 'fake_circuits.npy'), fake_circuits)
    np.save(os.path.join(results_path, 'sampled_cond.npy'), sampled_cond)


# Run simulation for each successful HPO
def generate_all_fake_circuits(df_hpos, datasets, input_species, postprocs: dict):
    successful_runs = df_hpos[df_hpos['run_successful'] | (df_hpos['R2_train'] > 0.8)]

    n_runs = len(successful_runs)

    fake_circuits_list = [None] * n_runs
    z_list = [None] * n_runs
    cond_list = [None] * n_runs
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
        cond_list[idx] = sampled_cond
        forward_rates_list[idx] = forward_rates
        reverse_rates_list[idx] = reverse_rates

    # Mixing all circuits together
    all_fake_circuits = np.concatenate(fake_circuits_list, axis=0)
    all_forward_rates = np.concatenate(forward_rates_list, axis=0)
    all_reverse_rates = np.concatenate(reverse_rates_list, axis=0)
    # all_z = dict(zip(successful_runs.index.tolist(), z_list))
    all_sampled_cond = dict(zip(successful_runs.index.tolist(), cond_list))

    del fake_circuits_list, z_list, cond_list, forward_rates_list, reverse_rates_list

    return all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond  # , all_z


def extend_analytics(analytics: dict, analytics_i: dict):
    for k in analytics_i.keys():
        if k not in analytics.keys():
            analytics[k] = analytics_i[k][None, :]
        else:
            analytics[k] = np.concatenate([analytics[k], analytics_i[k][None, :]])
    return analytics


def cvae_scan_multi(df_hpos: pd.DataFrame, fn_config_multisim: str, top_write_dir=TOP_WRITE_DIR, debug=False):
    """Run multiple CVAE scans and evaluate generated circuits"""
    os.makedirs(top_write_dir, exist_ok=True)

    # First run all models and save results
    df_hpos = loop_scans(df_hpos, top_write_dir, skip_verify=True, debug=debug)

    # Global options
    config_multisim = load_json_as_dict(fn_config_multisim)
    val_config = load_json_as_dict(config_multisim['filenames_train_config'])
    config_bio = {}
    for k in [kk for kk in val_config['base_configs_ensemble'].keys() if 'vis' not in kk]:
        config_bio.update(val_config['base_configs_ensemble'][k])

    # Pre-load datasets
    datasets = {k: load_by_fn(k)
                for k in df_hpos['filenames_train_table'].unique()}
    input_species = get_input_species(
        datasets[df_hpos['filenames_train_table'].unique()[0]])

    config_bio = prep_cfg(config_bio, input_species)
    model_brn, qreactions, postprocs, ordered_species = setup_model_brn(
        config_bio, input_species)

    # Simulate successful runs all in one go
    batch_size = config_multisim['eval_batch_size']
    all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond = generate_all_fake_circuits(
        df_hpos[df_hpos['run_successful']], datasets, input_species, postprocs)
    n_batches = int(np.ceil(len(all_fake_circuits) / batch_size))
    analytics, ys, ts, y0m, y00s, ts0 = {}, None, None, None, None, None
    start_time = datetime.now()
    for i in range(n_batches):
        i1, i2 = i*batch_size, (i+1)*batch_size
        logging.info(f'Simulating batch {i+1} of {n_batches} ({datetime.now() - start_time})')
        analytics_i, ys_i, ts_i, y0m_i, y00s_i, ts0_i = run_sim_multi(
            all_fake_circuits[i1:i2], all_forward_rates[i1:i2], all_reverse_rates[i1:i2], df_hpos['signal_species'].iloc[0], config_bio, model_brn, qreactions, ordered_species)
        analytics = extend_analytics(analytics, analytics_i)
        ys = np.concatenate([ys, ys_i[None, :]]) if ys is not None else ys_i[None, :]
        ts = np.concatenate([ts, ts_i[None, :]]) if ts is not None else ts_i[None, :]
        y0m = np.concatenate([y0m, y0m_i[None, :]]) if y0m is not None else y0m_i[None, :]
        y00s = np.concatenate([y00s, y00s_i[None, :]]) if y00s is not None else y00s_i[None, :]
        ts0 = np.concatenate([ts0, ts0_i[None, :]]) if ts0 is not None else ts0_i[None, :]
    
        # Save results
        save(top_write_dir, analytics, ys, ts, y0m,
            y00s, ts0, all_fake_circuits, all_sampled_cond)

    return df_hpos, analytics, ys, ts, y0m, y00s, ts0, all_fake_circuits, all_sampled_cond
