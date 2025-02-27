

import numpy as np
import os
import jax
import pandas as pd

from evoscaper.utils.preprocess import make_datetime_str
from evoscaper.utils.simulation import setup_model, make_rates, prep_sim, sim, prep_cfg
from evoscaper.utils.math import make_batch_symmetrical_matrices
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.data.fake_data_generation.energies import generate_energies
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, write_json


jax.config.update('jax_platform_name', 'gpu')


def save(returns_kwrgs, top_write_dir):
    for l, v in returns_kwrgs.items():
        ext = os.path.splitext(l)[1]
        if ext == '.npy':
            np.save(os.path.join(top_write_dir, l), v)
        elif ext == '.json':
            write_json(v, os.path.join(top_write_dir, l))
        else:
            print(f'Warning: could not save {l} of type {ext}')
    print(top_write_dir)


def simulate_interactions(interactions, input_species, config):

    if interactions.ndim == 2:
        interactions_reshaped = make_batch_symmetrical_matrices(
            interactions.reshape(-1, interactions.shape[-1]), side_length=len(input_species))
    else:
        interactions_reshaped = interactions

    model_brn, qreactions, ordered_species, postprocs = setup_model(
        interactions_reshaped, config, input_species, config['x_type'])

    forward_rates, reverse_rates = make_rates(
        config['x_type'], interactions_reshaped, postprocs)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, threshold_steady_states, total_time, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        config['signal_species'], qreactions, interactions_reshaped, config, forward_rates, reverse_rates)

    print('Starting sim')
    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller,
                                            threshold=threshold_steady_states,
                                            total_time=total_time)
    analytics['Log sensitivity'] = np.log10(
        analytics['sensitivity_wrt_species-6'])
    analytics['Log precision'] = np.log10(analytics['precision_wrt_species-6'])

    return analytics, ys, ts, y0m, y00s, ts0


def make_inputs(species_count, system_type):
    return [f'{system_type}_{i}' for i in range(species_count)]


def summarise_simulated_cicruits(analytics, config, interactions):
    n_species = config['circuit_generation']['species_count']
    data = pd.DataFrame.from_dict(
        {k: np.array(v)[..., -n_species:].flatten() for k, v in analytics.items()})
    data['sample_name'] = np.repeat(np.array(make_inputs(n_species, config['system_type']))[:, None],
                                    repeats=len(data) // n_species, axis=1).T.flatten()
    data['circuit_name'] = np.repeat(
        np.arange(len(data) // n_species), repeats=n_species)
    data['mutation_name'] = 'ref_circuit'
    data['mutation_num'] = 0
    data['mutation_type'] = '[]'
    data['mutation_positions'] = '[]'
    data['path_to_template_circuit'] = ''
    interactions_flat = jax.vmap(
        lambda x: x[np.triu_indices(n_species)])(interactions)
    idxs = np.array(np.triu_indices(n_species)).T
    idxs_same = np.where(idxs[:, 0] == idxs[:, 1])[0]
    idxs_diff = np.where(idxs[:, 0] != idxs[:, 1])[0]
    data['num_interacting'] = np.repeat(
        (interactions_flat != 0)[:, idxs_diff].sum(axis=1), repeats=n_species)
    data['num_self_interacting'] = np.repeat(
        (interactions_flat != 0)[:, idxs_same].sum(axis=1), repeats=n_species)
    for ii, (i, j) in enumerate(idxs):
        data[f'energies_{i}-{j}'] = np.repeat(
            interactions_flat[:, ii], repeats=n_species)
    return data


def main(top_write_dir=None, cfg_path=None):

    if top_write_dir is None:
        top_write_dir = os.path.join(
            'notebooks', 'data', 'simulate_circuits', make_datetime_str())

    data_dir = 'data'
    if cfg_path is None:
        config = {
            'signal_species': ['RNA_0'],
            'x_type': 'energies',
            'include_prod_deg': False,
            'system_type': 'RNA',
            'interaction_simulator': {'postprocess': True},
            'signal': {'function_kwargs': {'target': 2}},
            'molecular_params': {
                'avg_mRNA_per_cell': 100,
                'cell_doubling_time': 1200,
                'creation_rate': 2.35,
                'starting_copynumbers': 100,
                'degradation_rate': 0.01175,
                'association_binding_rate': 1000000
            },
            'circuit_generation': {
                'use_dataset': False,
                'dataset_src': f'{data_dir}/raw/summarise_simulation/2024_11_21_160955/tabulated_mutation_info.csv',
                'repetitions': 1000000,
                'species_count': 4,
                'sequence_length': 20,
                'generator_protocol': 'random',
                'proportion_to_mutate': 0.5,
                'template': None,
                'seed': 5
            },
            'simulation': {
                't0': 0,
                't1': 2000,
                'dt0': 0.001,
                'dt1': 0.5,
                'threshold_steady_states': 0.01,
                'total_time': 80000,
                'stepsize_controller': 'adaptive',
                'use_initial_to_add_signal': False,
            },
        }
    else:
        config_bio = load_json_as_dict(cfg_path)
        config = config_bio['base_configs_ensemble']['generate_species_templates']
        config.update(config_bio['base_configs_ensemble']
                      ['mutation_effect_on_interactions_signal'])

    input_species = make_inputs(
        config['circuit_generation']['species_count'], config['system_type'])

    config = prep_cfg(config, input_species)

    if config['circuit_generation']['use_dataset']:
        data = pd.read_csv(config['circuit_generation']['dataset_src'])
        n_species = len(data[~data['sample_name'].isna()]['sample_name'].unique())
        interactions = data[data['sample_name'] == 'RNA_2'][get_true_interaction_cols(data, 'energies', remove_symmetrical=False)].values.reshape(-1, n_species, n_species)
    else:
        interactions = generate_energies(
            n_circuits=config['circuit_generation'].get("repetitions", 1),
            n_species=config['circuit_generation'].get("species_count", 3), len_seq=config['circuit_generation']["sequence_length"],
            p_null=config['circuit_generation'].get("perc_non_interacting", 0.3),
            symmetrical=True if config.get(
                "system_type", "RNA") == 'RNA' else False,
            type_energies=config.get("system_type", "RNA"),
            seed=config['circuit_generation'].get("seed", 0))
    
    os.makedirs(top_write_dir, exist_ok=True)

    analytics, ys, ts, y0m, y00s, ts0 = simulate_interactions(
        interactions, input_species, config)

    save({'analytics.json': analytics,
          'ys.npy': ys, 'ts.npy': ts, 'y0m.npy': y0m, 'y00s.npy': y00s, 'ts0.npy': ts0,
          'interactions.npy': interactions,
          'config.json': config}, top_write_dir)

    data = summarise_simulated_cicruits(analytics, config, interactions)
    save({'tabulated_mutation_info.json': data}, top_write_dir)


if '__main__' == __name__:
    main()
