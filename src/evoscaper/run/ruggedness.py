

import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd
import os
import jax.numpy as jnp
import jax
from functools import partial
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, write_json
from evoscaper.scripts.verify import full_sim_prep
from evoscaper.scripts.cvae_scan import generate_all_fake_circuits
from evoscaper.utils.evolution import calculate_ruggedness_from_perturbations
from evoscaper.utils.math import make_flat_triangle
from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.simulation import make_rates, sim, load_config_bio, sim, prep_cfg, update_species_simulated_rates, setup_model_brn, compute_analytics


def calculate_ruggedness(interactions, eps_perc, analytic, input_species, x_type, signal_species, config_bio,
                         analytics_original: Optional[np.ndarray], resimulate_analytics: bool, top_write_dir: str):

    n_samples = interactions.shape[0]
    n_interactions = interactions.shape[1]
    n_perturbs = n_interactions + resimulate_analytics
    eps = eps_perc * np.abs(interactions).max()
    perturbations = jax.vmap(
        partial(create_perturbations, eps=eps))(interactions)
    if resimulate_analytics:
        perturbations = np.concatenate(
            [perturbations, interactions[:, None, :]], axis=1)

    analytics_perturbed, ys, ts, y0m, y00s, ts0 = simulate_perturbations(
        perturbations, x_type, signal_species, config_bio, input_species, top_write_dir)
    write_json(analytics_perturbed, os.path.join(
        top_write_dir, 'analytics.json'))

    analytic_perturbed = jnp.array(
        analytics_perturbed[analytic]).reshape(n_samples, n_perturbs, -1)
    if resimulate_analytics:
        analytic_perturbed = analytic_perturbed[:, :-1, :]
        analytic_og = analytic_perturbed[:, -1, :]
    elif analytics_original is not None:
        analytic_og = np.array(analytics_original[analytic][:n_samples])
    else:
        analytic_og = np.zeros_like(analytic_perturbed[:, -1, :])

    # If loaded from previous data where not all analytics were saved
    if analytic_perturbed.shape[-1] != analytic_og.shape[-1]:
        analytic_perturbed = analytic_perturbed[..., -analytic_og.shape[-1]:]

    ruggedness = jax.vmap(partial(calculate_ruggedness_from_perturbations, eps=eps))(
        analytic_perturbed, analytic_og[:, None, :])

    return ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0)


def create_perturbations(interactions, eps):

    interactions_expanded = jnp.ones(
        (len(interactions), len(interactions))) * interactions

    perturbations = interactions_expanded + \
        jnp.eye(len(interactions_expanded), len(interactions_expanded)) * eps

    return perturbations


def simulate_perturbations(interactions, x_type, signal_species, config_bio, input_species, top_write_dir: str):

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time,
        threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates,
        qreactions, model_brn) = full_sim_prep(interactions, input_species, x_type, signal_species, config_bio)

    #
    # threshold_steady_states = 0.01  # config_bio['simulation']['threshold_steady_states']
    # t1 = 1
    # total_time = 1
    #

    print('Starting sim')
    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller,
                                            threshold=threshold_steady_states,
                                            total_time=total_time)
    for i, l in zip([ys, ts, y0m, y00s, ts0], ['ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy']):
        np.save(os.path.join(top_write_dir, l), i)

    if (len(analytics) == 0):
        analytics = jax.vmap(partial(compute_analytics, t=ts, labels=np.arange(
            ys.shape[-1]), signal_onehot=signal_onehot))(ys)

    analytics['Log sensitivity'] = np.log10(analytics['sensitivity'])
    analytics['Log precision'] = np.log10(analytics['precision'])

    return analytics, ys, ts, y0m, y00s, ts0


def get_config_bio(config, fn_config_bio, input_species):
    config_bio = load_json_as_dict(fn_config_bio)
    config_bio_u = config_bio['base_configs_ensemble']['generate_species_templates']
    config_bio_u.update(
        config_bio['base_configs_ensemble']['mutation_effect_on_interactions_signal'])
    config_bio = prep_cfg(config_bio_u, input_species)

    config_sim = load_json_as_dict(config['fn_simulation_settings'])
    config_bio['simulation'].update(config_sim['simulation'])
    config_bio['simulation_steady_state'].update(
        config_sim['simulation_steady_state'])
    return config_bio


def verify_rugg(fake_circuits,
                config,
                config_bio,
                input_species,
                batch_size,
                top_write_dir,
                analytics_og=None):

    resimulate_analytics = config['resimulate_analytics']
    batch_size = int(np.ceil(batch_size / (fake_circuits.shape[-1] + resimulate_analytics)))
    n_batches = int(np.max([1, np.ceil(len(fake_circuits) / batch_size)]))
    eps_perc = config['eps_perc']
    x_type = config['x_type']
    signal_species = config['signal_species']
    analytic = config['analytic']
    eps = eps_perc * np.abs(fake_circuits).max()

    for i in range(1, n_batches):
        print(f'Batch {i}/{n_batches}')
        ii, ij = i * batch_size, (i + 1) * batch_size
        fake_circuits_batch = fake_circuits[ii:ij]
        top_write_dir_batch = os.path.join(top_write_dir, f'batch_{i}')
        os.makedirs(top_write_dir_batch, exist_ok=True)
        ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0) = calculate_ruggedness(
            fake_circuits_batch, eps_perc=eps_perc, analytic=analytic,
            input_species=input_species, x_type=x_type, signal_species=signal_species, config_bio=config_bio,
            analytics_original=analytics_og, resimulate_analytics=resimulate_analytics, top_write_dir=top_write_dir_batch)

        write_json(analytics_perturbed, os.path.join(
            top_write_dir, 'analytics.json'))
        for k, i in zip(['ruggedness.npy', 'ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy'], [ruggedness, ys, ts, y0m, y00s, ts0]):
            np.save(os.path.join(top_write_dir_batch, k), i)


def load_hpos(fn):
    try: 
        df_hpos = pd.read_json(fn)
    except ValueError:
        hpos = pd.Series({k: (v) if isinstance(v, list) else v for k, v in load_json_as_dict(fn_df_hpos_loaded).items()})
        df_hpos = pd.DataFrame(hpos).T
    if 'notebooks' in fn:
        df_hpos['filenames_train_table'] = df_hpos['filenames_train_table'].apply(lambda x: os.path.join('notebooks', x.replace('./', '')))
        df_hpos['filenames_verify_config'] = df_hpos['filenames_verify_config'].apply(lambda x: x.replace('../', ''))
    return df_hpos



def main(fn_df_hpos_loaded, config_run: dict):

    top_write_dir = os.path.join(
        'notebooks', 'data', 'ruggedness', make_datetime_str())
    os.makedirs(top_write_dir, exist_ok=True)
    
    df_hpos = load_hpos(fn_df_hpos_loaded)
    df_hpos['eval_n_to_sample'] = config_run['eval_n_to_sample']
    df_hpos['eval_cond_min'] = config_run['eval_cond_min']
    df_hpos['eval_cond_max'] = config_run['eval_cond_max']
    df_hpos['eval_n_categories'] = config_run['eval_n_categories']

    datasets = {v: pd.read_json(v) for v in df_hpos['filenames_train_table'].unique() if os.path.exists(v)}
    input_species = datasets[list(datasets.keys())[0]]['sample_name'].dropna().unique()
    fn_config_bio = df_hpos['filenames_verify_config'].dropna().unique()[0]
    config_bio = load_config_bio(fn_config_bio, input_species, config_run.get('fn_simulation_settings'))

    model_brn, qreactions, postprocs, ordered_species = setup_model_brn(
        config_bio, input_species)

    batch_size = config_run['eval_batch_size']
    all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond = generate_all_fake_circuits(
       df_hpos, datasets, input_species, postprocs)

    verify_rugg(make_flat_triangle(all_fake_circuits), config_run, config_bio,
                input_species, batch_size, top_write_dir, analytics_og=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_df_hpos_loaded', type=str, default=None,
                        help='Path to dataframe of hyperparameters and results from previous run (json).')
    args = parser.parse_args()
    fn_df_hpos_loaded = args.fn_df_hpos_loaded
    # fn_df_hpos_loaded = 'notebooks/data/01_cvae/2025_03_12__16_14_02/saves_2025_03_12__16_14_02_ds0211_srugg_hs32_nl3_KL2e4_cont01ts095pd3_lr1e3_teva97'
    fn_df_hpos_loaded = 'notebooks/data/01_cvae/2025_03_12__16_14_02/hpos_all.json'

    config_run = {
        'eps_perc': -1e-2,
        'x_type': 'energies',
        'signal_species': 'RNA_0',
        'resimulate_analytics': True,
        'analytic': 'Log sensitivity',
        'eval_batch_size': int(3e5),
        'eval_n_to_sample': int(1e5),
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,   
        'eval_n_categories': 10,
        'fn_simulation_settings': 'notebooks/configs/cvae_multi/simulation_settings.json'
    }

    main(fn_df_hpos_loaded, config_run)
