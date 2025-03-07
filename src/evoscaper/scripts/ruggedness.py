

from typing import Optional
import numpy as np
import os
import jax.numpy as jnp
import jax
from functools import partial
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict, write_json
from evoscaper.scripts.verify import full_sim_prep
from evoscaper.utils.simulation import prep_cfg, sim, compute_analytics
from evoscaper.utils.evolution import calculate_ruggedness_from_perturbations


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
    else:
        analytic_og = np.array(analytics_original[analytic][:n_samples])

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

    calculate_analytics = True

    print('Starting sim')
    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller,
                                            threshold=threshold_steady_states,
                                            total_time=total_time,
                                            calculate_analytics=calculate_analytics)
    for i, l in zip([ys, ts, y0m, y00s, ts0], ['ys.npy', 'ts.npy', 'y0m.npy', 'y00s.npy', 'ts0.npy']):
        np.save(os.path.join(top_write_dir, l), i)

    if not calculate_analytics and (len(analytics) == 0):
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
                fn_config_bio,
                input_species,
                top_write_dir,
                analytics_og=None):

    config_bio = get_config_bio(config, fn_config_bio, input_species)
    eps_perc = config['eps_perc']
    x_type = config['x_type']
    signal_species = config['signal_species']
    resimulate_analytics = config['resimulate_analytics']
    eps = eps_perc * np.abs(fake_circuits).max()
    print('eps:', eps)

    ruggedness, (analytics_perturbed, ys, ts, y0m, y00s, ts0) = calculate_ruggedness(fake_circuits, eps_perc=eps_perc, analytic='Log sensitivity',
                                                                                     input_species=input_species, x_type=x_type, signal_species=signal_species, config_bio=config_bio,
                                                                                     analytics_original=analytics_og, resimulate_analytics=resimulate_analytics, top_write_dir=top_write_dir)


def main():

    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=config_norm_y.categorical_n_bins if config_norm_y.categorical_n_bins else 10,
                                                            n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=use_binned_sampling, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max(), clip_range=impose_final_range)
