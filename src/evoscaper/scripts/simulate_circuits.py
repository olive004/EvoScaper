

import numpy as np
import os

from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from evoscaper.utils.simulation import setup_model, make_rates, prep_sim, sim, prep_cfg
from evoscaper.utils.math import make_batch_symmetrical_matrices
from synbio_morpher.utils.data.fake_data_generation.energies import generate_energies
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, write_json


def save(returns_kwrgs, top_write_dir):
    for l, v in returns_kwrgs.items():
        ext = os.path.splitext(l)
        if ext == 'npy':
            np.save(os.path.join(top_write_dir, l), v)
        elif ext == 'json':
            write_json(v, os.path.join(top_write_dir, 'analytics.json'))
    print(top_write_dir)


def simulate_interactions(interactions, input_species, config):

    interactions_reshaped = make_batch_symmetrical_matrices(
        interactions.reshape(-1, interactions.shape[-1]), side_length=len(input_species))

    model_brn, qreactions, ordered_species, postproc = setup_model(
        interactions_reshaped, config, input_species)

    forward_rates, reverse_rates = make_rates(
        config['x_type'], interactions_reshaped, postproc)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        config['signal_species'], qreactions, interactions_reshaped, config, forward_rates, reverse_rates)

    #
    threshold = 0.005  # config['simulation']['threshold_steady_states']
    t1 = 500
    total_time = 30000
    #

    print('Starting sim')
    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller,
                                            threshold=threshold,
                                            total_time=total_time)
    analytics['Log sensitivity'] = np.log10(
        analytics['sensitivity_wrt_species-6'])
    analytics['Log precision'] = np.log10(analytics['precision_wrt_species-6'])

    return analytics, ys, ts, y0m, y00s, ts0


def make_inputs(species_count, system_type):
    return [f'{system_type}_{i}' for i in range(species_count)]


def main(top_write_dir, cfg_path=None):

    if cfg_path is None:
        config = {
            'signal_species': ['RNA_0'],
            'x_type': 'energies',
            'include_prod_deg': False,
            'system_type': 'RNA',
            'molecular_params': {
                'avg_mRNA_per_cell': 100,
                'cell_doubling_time': 1200,
                'creation_rate': 2.35,
                'starting_copynumbers': 100,
                'degradation_rate': 0.01175,
                'association_binding_rate': 1000000
            },
            'circuit_generation': {
                'repetitions': 700000,
                'species_count': 3,
                'sequence_length': 20,
                'generator_protocol': 'random',
                'proportion_to_mutate': 0.5,
                'template': None,
                'seed': 4
            },
            'simulation': {
                't0': 0,
                't1': 1000,
                'dt0': 0.1,
                'dt1': 0.1,
                'threshold_steady_states': 0.001,
                'total_time': 30000
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
    interactions = generate_energies(**config['circuit_generation'])

    os.make_dirs(top_write_dir, exist_ok=True)

    analytics, ys, ts, y0m, y00s, ts0 = simulate_interactions(
        interactions, input_species, config)
    save({'analytics.json': analytics,
          'ys.npy': ys, 'ts.npy': ts, 'y0m.npy': y0m, 'y00s.npy': y00s, 'ts0.npy': ts0,
          'config.json': config}, top_write_dir)
