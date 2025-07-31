

import logging
import argparse
from typing import Optional
import numpy as np
import pandas as pd
import os
import jax.numpy as jnp
import jax
from functools import partial
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, write_json
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from evoscaper.scripts.verify import full_sim_prep
from evoscaper.scripts.cvae_scan import generate_all_fake_circuits
from evoscaper.utils.evolution import calculate_ruggedness_core
from evoscaper.utils.math import make_flat_triangle
from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.simulation import load_config_bio, sim, prep_cfg, setup_model_brn, compute_analytics

jax.config.update('jax_platform_name', 'gpu')


def calculate_ruggedness(interactions, eps_perc: float, analytic, input_species, x_type, signal_species, config_bio,
                         analytics_original: Optional[np.ndarray], resimulate_analytics: bool, top_write_dir: str,
                         n_perturbs: int = 1, perturb_once: bool = True):

    n_samples = interactions.shape[0]

    perturbations, eps, n_perturbs = get_perturbations(
        interactions, eps_perc, n_samples, n_perturbs, resimulate_analytics, perturb_once)
    np.save(os.path.join(top_write_dir, 'eps.npy'), eps)

    ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0) = sim_save_rugg(
        perturbations, x_type, signal_species, config_bio, input_species, top_write_dir,
        analytics_original, analytic, resimulate_analytics, n_samples, n_perturbs, eps, perturb_once)

    return ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0)


def get_perturbations(interactions, eps_perc, n_samples, n_perturbs, resimulate_analytics, perturb_once,
                      seed=0):

    n_interactions = interactions.shape[1]

    if perturb_once:
        eps = eps_perc * np.abs(interactions).max()
        n_perturbs = n_interactions + resimulate_analytics
        perturbations = jax.vmap(
            partial(create_perturbations, eps=eps))(interactions)
    else:
        eps = jax.random.uniform(
            jax.random.PRNGKey(seed),
            (n_samples, n_perturbs, n_interactions)) * eps_perc * np.abs(interactions).max()
        perturbations = interactions[:, None, :] + eps
        n_perturbs = n_perturbs + resimulate_analytics

    perturbations = add_original(
        perturbations, interactions, resimulate_analytics)

    return perturbations, eps, n_perturbs


def sim_save_rugg(perturbations, x_type, signal_species, config_bio, input_species, top_write_dir,
                  analytics_original, analytic, resimulate_analytics, n_samples, n_perturbs, eps, perturb_once: bool):

    analytics_perturbed, ys, ts, y0m, y00s, ts0 = simulate_perturbations(
        perturbations, x_type, signal_species, config_bio, input_species, top_write_dir)
    write_json(analytics_perturbed, os.path.join(
        top_write_dir, 'analytics.json'))

    ruggedness = calculate_ruggedness_core(analytics_perturbed, analytics_original, analytic,
                                           resimulate_analytics, n_samples, n_perturbs, eps, perturb_once)
    return ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0)


def add_original(perturbations, interactions, resimulate_analytics):
    if resimulate_analytics:
        perturbations = np.concatenate(
            [perturbations, interactions[:, None, :]], axis=1)
    return perturbations


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

    logging.info('Starting sim')
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


def simulate_rugg(fake_circuits,
                  config_bio,
                  input_species,
                  batch_size,
                  eps_perc,
                  x_type,
                  signal_species,
                  analytic,
                  top_write_dir,
                  resimulate_analytics=True,
                  perturb_once=True,
                  n_perturbs=1,
                  analytics_og=None):

    batch_size = int(
        np.ceil(batch_size / (fake_circuits.shape[-1] + resimulate_analytics)))
    n_batches = int(np.max([1, np.ceil(len(fake_circuits) / batch_size)]))

    for i in range(n_batches):
        logging.info(f'Batch {i+1}/{n_batches}')
        ii, ij = i * batch_size, (i + 1) * batch_size
        fake_circuits_batch = fake_circuits[ii:ij]
        top_write_dir_batch = os.path.join(top_write_dir, f'batch_{i}')
        os.makedirs(top_write_dir_batch, exist_ok=True)
        ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0) = calculate_ruggedness(
            fake_circuits_batch, eps_perc=eps_perc, analytic=analytic,
            input_species=input_species, x_type=x_type, signal_species=signal_species, config_bio=config_bio,
            analytics_original=analytics_og, resimulate_analytics=resimulate_analytics, top_write_dir=top_write_dir_batch,
            n_perturbs=n_perturbs, perturb_once=perturb_once)

        write_json(analytics_perturbed, os.path.join(
            top_write_dir, 'analytics.json'))
        for k, ii in zip(['ruggedness.npy', 'ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy'], [ruggedness, ys, ts, y0m, y00s, ts0]):
            np.save(os.path.join(top_write_dir_batch, k), ii)


def load_hpos(fn):
    try:
        df_hpos = pd.read_json(fn)
    except ValueError:
        hpos = pd.Series({k: (v) if isinstance(
            v, list) else v for k, v in load_json_as_dict(fn_df_hpos_loaded).items()})
        df_hpos = pd.DataFrame(hpos).T
    if 'notebooks' in fn:
        df_hpos['filenames_train_table'] = df_hpos['filenames_train_table'].apply(
            lambda x: os.path.join('notebooks', x.replace('./', '')))
        df_hpos['filenames_verify_config'] = df_hpos['filenames_verify_config'].apply(
            lambda x: x.replace('../', ''))
    return df_hpos


def run_circuits(fn_circuits, config_run, top_write_dir: str):
    """ Simulate the ruggedness just for the circuits loaded. """

    if type(fn_circuits) == str:
        fn_circuits = [fn_circuits]
    circuits = np.concatenate([np.load(fn)[None, ...]
                              for fn in fn_circuits], axis=0)
    input_species = config_run['input_species']
    config_bio = load_config_bio(
        config_run['filenames_verify_config'], input_species, config_run.get('fn_simulation_settings'))
    config_bio['simulation']['save_steps'] = 500
    # config_bio['simulation']['dt0'] = 0.0005
    config_bio['simulation']['t1'] = 500
    # config_bio['simulation']['threshold_steady_states'] = 0.005
    simulate_rugg(circuits,
                  config_bio,
                  input_species,
                  config_run['eval_batch_size'],
                  config_run['eps_perc'],
                  config_run['x_type'],
                  config_run['signal_species'],
                  config_run['analytic'],
                  top_write_dir=top_write_dir,
                  resimulate_analytics=config_run['resimulate_analytics'],
                  perturb_once=config_run['perturb_once'],
                  n_perturbs=config_run['n_perturbs'],
                  analytics_og=None)


def run_dataset(fn_ds, config_run, top_write_dir: str):

    data = pd.read_json(fn_ds) if fn_ds.endswith(
        '.json') else pd.read_csv(fn_ds)
    input_species = data['sample_name'].dropna().unique()
    config_bio = load_config_bio(
        config_run['filenames_verify_config'], input_species, config_run.get('fn_simulation_settings'))

    circuits = data[data['sample_name'] == input_species[0]][get_true_interaction_cols(
        data, 'energies', remove_symmetrical=True)].values
    simulate_rugg(circuits,
                  config_bio,
                  input_species,
                  config_run['eval_batch_size'],
                  config_run['eps_perc'],
                  config_run['x_type'],
                  config_run['signal_species'],
                  config_run['analytic'],
                  top_write_dir=top_write_dir,
                  resimulate_analytics=config_run['resimulate_analytics'],
                  perturb_once=config_run['perturb_once'],
                  n_perturbs=config_run['n_perturbs'],
                  analytics_og=None)


def main(fn_df_hpos_loaded, config_run: dict, top_write_dir: str):

    df_hpos = load_hpos(fn_df_hpos_loaded)
    df_hpos['eval_n_to_sample'] = config_run['eval_n_to_sample']
    cond_min = [tuple(config_run['eval_cond_min'])] * len(df_hpos) if type(
        config_run['eval_cond_min']) in [list, tuple] else config_run['eval_cond_min']
    cond_max = [tuple(config_run['eval_cond_max'])] * len(df_hpos) if type(
        config_run['eval_cond_max']) in [list, tuple] else config_run['eval_cond_max']
    df_hpos['eval_cond_min'] = cond_min
    df_hpos['eval_cond_max'] = cond_max
    df_hpos['eval_n_categories'] = config_run['eval_n_categories']

    datasets = {v: pd.read_json(
        v) for v in df_hpos['filenames_train_table'].unique() if os.path.exists(v)}
    input_species = sorted(datasets[list(datasets.keys())[
        0]]['sample_name'].dropna().unique())
    fn_config_bio = df_hpos['filenames_verify_config'].dropna().unique()[0]
    config_bio = load_config_bio(
        fn_config_bio, input_species, config_run.get('fn_simulation_settings'))

    model_brn, qreactions, postprocs, ordered_species = setup_model_brn(
        config_bio, input_species)

    all_fake_circuits, all_forward_rates, all_reverse_rates, all_sampled_cond = generate_all_fake_circuits(
        df_hpos, datasets, input_species, postprocs)

    np.save(os.path.join(top_write_dir, 'all_fake_circuits.npy'), all_fake_circuits)
    if type(all_sampled_cond) == dict:
        os.makedirs(os.path.join(top_write_dir, 'sampled_cond'), exist_ok=True)
        for i in all_sampled_cond.keys():
            np.save(os.path.join(top_write_dir, 'sampled_cond',
                    f'sampled_cond_{i}'), all_sampled_cond[i])
        # [os.makedirs(os.path.join(top_write_dir, str(k)), exist_ok=True) for k, v in all_sampled_cond.items()]
        # [np.save(os.path.join(top_write_dir, str(k), 'all_sampled_cond.npy'), v) for k, v in all_sampled_cond.items()]
    else:
        np.save(os.path.join(top_write_dir,
                'all_sampled_cond.npy'), all_sampled_cond)
    simulate_rugg(make_flat_triangle(all_fake_circuits),
                  config_bio,
                  input_species,
                  config_run['eval_batch_size'],
                  config_run['eps_perc'],
                  config_run['x_type'],
                  config_run['signal_species'],
                  config_run['analytic'],
                  top_write_dir=top_write_dir,
                  resimulate_analytics=config_run['resimulate_analytics'],
                  perturb_once=config_run['perturb_once'],
                  n_perturbs=config_run['n_perturbs'],
                  analytics_og=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_df_hpos_loaded', type=str, default=None,
                        help='Path to dataframe of hyperparameters and results from previous run (json).')
    parser.add_argument('--fn_ds', type=str, default=None,
                        help='Path to dataset that you want to simulate directly on its own.')
    parser.add_argument('--fn_circuits', type=str, default=None,
                        help='Path to dataset that you want to simulate directly on its own.')
    args = parser.parse_args()
    fn_df_hpos_loaded = args.fn_df_hpos_loaded
    fn_ds = args.fn_ds
    fn_circuits = args.fn_circuits

    fn_saves = None
    fn_saves = 'notebooks/data/01_cvae/2025_03_25__12_04_01/saves_2025_03_25__12_04_01_ds0211_arugg_hs32_nl3_KL2e4_cont01ts095pd3_lr1e3_teva98'
    fn_df_hpos_loaded = 'notebooks/data/01_cvae/2025_03_25__12_04_01/hpos_all.json'
    fn_ds = None  # 'notebooks/data/simulate_circuits/2025_01_29__18_12_38/tabulated_mutation_info.json'
    fn_circuits = None
    fn_circuits = ['notebooks/data/16_visualise_rugged_verify/2025_07_31__09_47_30/circuit_chosen_rugg_hi.npy',
       'notebooks/data/16_visualise_rugged_verify/2025_07_31__09_47_30/circuit_chosen_rugg_lo.npy']
    /workdir/notebooks/data/16_visualise_rugged_verify/2025_07_31__09_47_30
    if fn_df_hpos_loaded is not None:
        try:
            fn_saves = [i for i in os.listdir(os.path.dirname(
                fn_df_hpos_loaded)) if i.startswith('save')][0]
        except:
            pass

    # Run circuits individually
    config_run = {
        'eps_perc': 0.1,
        'x_type': 'energies',
        'signal_species': 'RNA_0',
        'resimulate_analytics': True,
        'perturb_once': False,
        'n_perturbs': int(2e4),
        'analytic': 'Log sensitivity',
        'eval_batch_size': int(1e5),
        'eval_n_to_sample': int(1e5),
        'eval_cond_min': -0.2,
        'eval_cond_max': 1.2,
        'eval_n_categories': 10,
        'fn_saves': fn_saves,
        'fn_df_hpos_loaded': fn_df_hpos_loaded,
        'fn_ds': fn_ds,
        'fn_circuits': fn_circuits,
        'fn_simulation_settings': 'notebooks/configs/cvae_multi/simulation_settings2.json',
        # 'filenames_verify_config': 'data/raw/summarise_simulation/2024_12_05_210221/ensemble_config_update.json'
        # 'filenames_verify_config': 'notebooks/data/simulate_circuits/2025_01_29__18_12_38/config.json',
        'filenames_verify_config': 'data/raw/summarise_simulation/2024_11_27_145142/ensemble_config.json',
        'input_species': ['RNA_0', 'RNA_1', 'RNA_2']  # None,
    }
    
    config_run = {
        "eps_perc": -0.01,
        "eval_cond_min": [-0.1, -0.1],
        "eval_cond_max": [1.1, 1.1],
        "eval_n_categories": 13,
    }

    # Run model-generated circuits
    # config_run = {
    #     "eps_perc": -0.01,
    #     "x_type": "energies",
    #     "signal_species": "RNA_0",
    #     "resimulate_analytics": True,
    #     'perturb_once': True,
    #     'n_perturbs': 1,
    #     "analytic": "Log sensitivity",
    #     "eval_batch_size": int(1e5),
    #     "eval_n_to_sample": int(1500000), #int(1e6),
    #     "eval_cond_min": [-0.1, -0.1],
    #     "eval_cond_max": [1.1, 1.1],
    #     "eval_n_categories": 13,
    #     "fn_saves": None,
    #     "fn_df_hpos_loaded": fn_df_hpos_loaded,
    #     "fn_ds": fn_ds,
    #     'fn_circuits': fn_circuits,
    #     "fn_simulation_settings": "notebooks/configs/cvae_multi/simulation_settings2.json",
    #     "filenames_verify_config": "data/raw/summarise_simulation/2024_11_27_145142/ensemble_config.json"
    # }

    top_write_dir = os.path.join(
        'notebooks', 'data', 'ruggedness', make_datetime_str())
    os.makedirs(top_write_dir, exist_ok=True)

    write_json(config_run, os.path.join(top_write_dir, 'config.json'))

    if fn_ds is not None:
        logging.info(f'Simulating ruggedness only for dataset {fn_ds}')
        run_dataset(fn_ds, config_run, top_write_dir)
    elif fn_df_hpos_loaded is not None:
        logging.info(f'Simulating ruggedness for model {fn_df_hpos_loaded}')
        main(fn_df_hpos_loaded, config_run, top_write_dir)
    elif fn_circuits is not None:
        logging.info(f'Simulating ruggedness for circuits {fn_circuits}')
        run_circuits(fn_circuits, config_run, top_write_dir)
    else:
        logging.info('Specify either fn_df_hpos_loaded or fn_ds')
