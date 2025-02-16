

import os
from typing import List
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation
from synbio_morpher.utils.results.result_writer import ResultWriter
from synbio_morpher.srv.io.manage.script_manager import script_preamble

import numpy as np
import jax
import pandas as pd

from evoscaper.model.evaluation import calc_prompt_adherence
from evoscaper.model.sampling import sample_reconstructions
from evoscaper.utils.dataclasses import DatasetConfig, ModelConfig, NormalizationSettings
from evoscaper.utils.math import make_batch_symmetrical_matrices
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.simulation import setup_model, make_rates, prep_sim, sim, prep_cfg
from evoscaper.utils.visualise import vis_sampled_histplot


jax.config.update('jax_platform_name', 'gpu')


def save(data_writer, analytics, ys, ts, y0m, y00s, ts0, fake_circuits, sampled_cond):
    print(data_writer.top_write_dir)
    data_writer.output(data=analytics, out_type='json', out_name='analytics')
    data_writer.output(data=ys, out_type='npy', out_name='ys')
    data_writer.output(data=ts, out_type='npy', out_name='ts')
    data_writer.output(data=y0m, out_type='npy', out_name='y0m')
    data_writer.output(data=y00s, out_type='npy', out_name='y0m')
    data_writer.output(data=ts0, out_type='npy', out_name='y0m')
    data_writer.output(data=fake_circuits, out_type='npy',
                       out_name='fake_circuits')
    data_writer.output(data=sampled_cond, out_type='npy',
                       out_name='sampled_cond')


def verify(params, rng, decoder,
           df: pd.DataFrame, cond,
           config_bio: dict,
           config_norm_y: NormalizationSettings,
           config_dataset: DatasetConfig,
           config_model: ModelConfig,
           x_datanormaliser: DataNormalizer, x_methods_preprocessing: List[str],
           y_datanormaliser: DataNormalizer,
           output_species: List[str],
           signal_species: List[str],
           input_species: List[str],
           n_to_sample: int = 100000,
           visualise=True,
           top_write_dir=None,
           return_relevant=False,
           data_writer=None,
           impose_final_range=None,
           use_binned_sampling: bool = True):

    if top_write_dir is not None:
        data_writer = ResultWriter(
            purpose=config_bio.get('experiment', {}).get('purpose', 'ensemble_simulate_by_interaction'), out_location=top_write_dir)
    config_bio, data_writer = script_preamble(
        config_bio, data_writer=data_writer)

    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=config_norm_y.categorical_n_bins if config_norm_y.categorical_n_bins else 10,
                                                            n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=use_binned_sampling, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max(), impose_final_range=impose_final_range)

    # input_species = df[df['sample_name'].notna()]['sample_name'].unique()

    fake_circuits_reshaped = make_batch_symmetrical_matrices(
        fake_circuits.reshape(-1, fake_circuits.shape[-1]), side_length=len(input_species))

    config_bio = prep_cfg(config_bio, input_species)
    model_brn, qreactions, ordered_species, postproc = setup_model(
        fake_circuits_reshaped, config_bio, input_species)

    forward_rates, reverse_rates = make_rates(
        config_dataset.x_type, fake_circuits_reshaped, postproc)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, total_time,
     threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        signal_species, qreactions, fake_circuits_reshaped, config_bio, forward_rates, reverse_rates)

    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller, threshold=threshold_steady_states,
                                            total_time=total_time)

    analytics['sensitivity_wrt_species-6'] = np.array(
        analytics['sensitivity_wrt_species-6'])
    analytics['precision_wrt_species-6'] = np.array(
        analytics['precision_wrt_species-6'])
    analytics['overshoot'] = np.array(analytics['overshoot'])
    analytics['Log sensitivity'] = np.log10(
        analytics['sensitivity_wrt_species-6'])
    analytics['Log precision'] = np.log10(analytics['precision_wrt_species-6'])

    if visualise:
        all_species = list([ii.name for ii in model_brn.species])
        idx_obj = 0 if 'Log sensitivity' not in config_dataset.objective_col else config_dataset.objective_col.index(
            'Log sensitivity')
        vis_sampled_histplot(analytics['sensitivity_wrt_species-6'], all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='layer', save_path=os.path.join(data_writer.top_write_dir, 'sens_layer.png'))
        vis_sampled_histplot(analytics['sensitivity_wrt_species-6'], all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='fill', save_path=os.path.join(data_writer.top_write_dir, 'sens_fill.png'))
        vis_sampled_histplot(calculate_adaptation(analytics['sensitivity_wrt_species-6'], analytics['precision_wrt_species-6']), all_species, output_species, category_array=sampled_cond[..., idx_obj].reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Adaptation of generated circuits', x_label=f'Adaptation to signal {signal_species}', multiple='layer', save_path=os.path.join(data_writer.top_write_dir, 'adapt_layer.png'))
    save(data_writer, analytics, ys, ts, y0m,
         y00s, ts0, fake_circuits, sampled_cond)

    # precision, recall = calc_prompt_adherence(sampled_cond, np.concatenate(
    #     [np.array(analytics[k])[:, None] for k in config_dataset.objective_col], axis=-1).reshape(*sampled_cond.shape, -1), perc_recall=0.1)
    # adh = {'precision': precision, 'recall': recall, 'diff_m': diff_m, 'diff_s': diff_s}
    # data_writer.output(data=adh, out_type='json', out_name='recall')

    if return_relevant:
        return analytics, ys, ts, y0m, y00s, ts0, fake_circuits, reverse_rates, model_brn, qreactions, ordered_species, input_species, z, sampled_cond
