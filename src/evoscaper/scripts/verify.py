

import os
from typing import List
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation
from synbio_morpher.utils.data.data_format_tools.common import write_json

import numpy as np
import jax
import pandas as pd

from evoscaper.model.sampling import sample_reconstructions
from evoscaper.utils.dataclasses import DatasetConfig, ModelConfig, NormalizationSettings
from evoscaper.utils.math import make_batch_symmetrical_matrices
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.simulation import setup_model, make_rates, prep_sim, sim, prep_cfg
from evoscaper.utils.visualise import vis_sampled_histplot


jax.config.update('jax_platform_name', 'gpu')


def save(top_write_dir, analytics, ys, ts, y0m, y00s, ts0, fake_circuits, sampled_cond):
    print(top_write_dir)
    write_json(analytics, os.path.join(top_write_dir, 'analytics.json'))
    np.save(os.path.join(top_write_dir, 'ys.npy'), ys)
    np.save(os.path.join(top_write_dir, 'ts.npy'), ts)
    np.save(os.path.join(top_write_dir, 'y0m.npy'), y0m)
    np.save(os.path.join(top_write_dir, 'y00s.npy'), y00s)
    np.save(os.path.join(top_write_dir, 'ts0.npy'), ts0)
    np.save(os.path.join(top_write_dir, 'fake_circuits.npy'), fake_circuits)
    np.save(os.path.join(top_write_dir, 'sampled_cond.npy'), sampled_cond)


def full_sim_prep(fake_circuits, input_species, x_type: str, signal_species, config_bio):
    fake_circuits_reshaped = make_batch_symmetrical_matrices(
        fake_circuits.reshape(-1, fake_circuits.shape[-1]), side_length=len(input_species))

    config_bio = prep_cfg(config_bio, input_species)
    model_brn, qreactions, ordered_species, postprocs = setup_model(
        fake_circuits_reshaped, config_bio, input_species, x_type)

    forward_rates, reverse_rates = make_rates(
        x_type, fake_circuits_reshaped, postprocs)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time,
     threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        signal_species, qreactions, fake_circuits_reshaped, config_bio, forward_rates, reverse_rates)

    return (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time,
            threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates,
            qreactions, model_brn)


def verify(params, rng, decoder,
           cond,
           config_bio: dict,
           config_norm_y: NormalizationSettings,
           config_dataset: DatasetConfig,
           config_model: ModelConfig,
           x_datanormaliser: DataNormalizer, x_methods_preprocessing: List[str],
           output_species: List[str],
           signal_species: List[str],
           input_species: List[str],
           top_write_dir: str,
           n_to_sample: int = 100000,
           visualise=True,
           return_relevant=False,
           impose_final_range=None,
           use_binned_sampling: bool = True):

    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=config_norm_y.categorical_n_bins if config_norm_y.categorical_n_bins else 10,
                                                            n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=use_binned_sampling, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max(), clip_range=impose_final_range)

    # input_species = df[df['sample_name'].notna()]['sample_name'].unique()

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time,
        threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates,
        qreactions, model_brn) = full_sim_prep(fake_circuits, input_species, config_dataset.x_type, signal_species, config_bio)

    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller, threshold=threshold_steady_states,
                                            total_time=total_time)

    analytics['sensitivity'] = np.array(
        analytics['sensitivity'])
    analytics['precision'] = np.array(
        analytics['precision'])
    analytics['overshoot'] = np.array(analytics['overshoot'])
    analytics['Log sensitivity'] = np.log10(
        analytics['sensitivity'])
    analytics['Log precision'] = np.log10(analytics['precision'])

    if visualise:
        all_species = list([ii.name for ii in model_brn.species])
        idx_obj = 0 if 'Log sensitivity' not in config_dataset.objective_col else config_dataset.objective_col.index(
            'Log sensitivity')
        vis_sampled_histplot(analytics['sensitivity'], all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='layer', save_path=os.path.join(top_write_dir, 'sens_layer.png'))
        vis_sampled_histplot(analytics['sensitivity'], all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='fill', save_path=os.path.join(top_write_dir, 'sens_fill.png'))
        vis_sampled_histplot(calculate_adaptation(analytics['sensitivity'], analytics['precision']), all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Adaptation of generated circuits', x_label=f'Adaptation to signal {signal_species}', multiple='layer', save_path=os.path.join(top_write_dir, 'adapt_layer.png'))
    save(top_write_dir, analytics, ys, ts, y0m,
         y00s, ts0, fake_circuits, sampled_cond)

    # precision, recall = calc_prompt_adherence(sampled_cond, np.concatenate(
    #     [np.array(analytics[k])[:, None] for k in config_dataset.objective_col], axis=-1).reshape(*sampled_cond.shape, -1), perc_recall=0.1)
    # adh = {'precision': precision, 'recall': recall, 'diff_m': diff_m, 'diff_s': diff_s}
    # data_writer.output(data=adh, out_type='json', out_name='recall')

    if return_relevant:
        return analytics, ys, ts, y0m, y00s, ts0, fake_circuits, reverse_rates, model_brn, qreactions, input_species, z, sampled_cond
