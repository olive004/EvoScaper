

import os
from typing import List
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.srv.parameter_prediction.simulator import RawSimulationHandling
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
from synbio_morpher.utils.common.setup import prepare_config, expand_config, expand_model_config
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.results.analytics.naming import get_true_interaction_cols
from synbio_morpher.utils.results.analytics.timeseries import calculate_adaptation
from synbio_morpher.utils.misc.type_handling import flatten_listlike, get_unique
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from synbio_morpher.utils.modelling.solvers import simulate_steady_states, make_stepsize_controller
from synbio_morpher.utils.results.result_writer import ResultWriter
from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.model.data_containers import BasicModel, QuantifiedReactions
from functools import partial

import numpy as np
import jax
import diffrax as dfx
import pandas as pd

from evoscaper.model.sampling import sample_reconstructions
from evoscaper.utils.dataclasses import DatasetConfig, ModelConfig, NormalizationSettings
from evoscaper.utils.math import make_batch_symmetrical_matrices, make_symmetrical_matrix_from_sequence_nojax
from evoscaper.utils.normalise import DataNormalizer
from evoscaper.utils.visualise import vis_sampled_histplot


jax.config.update('jax_platform_name', 'gpu')


def update_species_simulated_rates(ordered_species: list, forward_interactions, reverse_interactions, model: BasicModel, qreactions: QuantifiedReactions):
    for i, r in enumerate(model.reactions):
        if len(r.input) == 2:
            model.reactions[i].forward_rate = forward_interactions[
                ordered_species.index(r.input[0]), ordered_species.index(r.input[1])]

            model.reactions[i].reverse_rate = reverse_interactions[
                ordered_species.index(r.input[0]), ordered_species.index(r.input[1])]
    qreactions.reactions = qreactions.init_reactions(
        model)
    return model, qreactions


def compute_analytics(y, t, labels, signal_onehot):
    y = np.swapaxes(y, 0, 1)

    analytics_func = partial(
        generate_analytics, time=t, labels=labels,
        signal_onehot=signal_onehot, signal_time=t[1],
        ref_circuit_data=None)
    return analytics_func(data=y, time=t, labels=labels)


def make_rates(x_type, fake_circuits_reshaped, postproc):
    if x_type == 'energies':
        eqconstants, (forward_rates, reverse_rates) = postproc(
            fake_circuits_reshaped)
    elif x_type == 'binding_rates_dissociation':
        reverse_rates = fake_circuits_reshaped
        # eqconstants = forward_rates[0, 0, 0] / reverse_rates
    else:
        raise ValueError(f'Unknown x_type {x_type}')
    return forward_rates, reverse_rates


def sim(y00, forward_rates, reverse_rates,
        qreactions,
        signal_onehot, signal_target,
        t0, t1, dt0, dt1,
        save_steps, max_steps,
        stepsize_controller,
        dt1_factor=5,
        threshold=0.01):
    """ Concentrations should be in the form [circuits, time, species] """

    rate_max = np.max([np.max(forward_rates),
                       np.max(reverse_rates)])
    dt0 = np.min([1 / (5 * rate_max), dt0])
    dt1 = dt1_factor * dt0

    sim_func = jax.jit(jax.vmap(partial(bioreaction_sim_dfx_expanded,
                                t0=t0, t1=t1, dt0=dt0,
                                signal=None, signal_onehot=None,
                                inputs=qreactions.reactions.inputs,
                                outputs=qreactions.reactions.outputs,
                                forward_rates=forward_rates,
                                solver=dfx.Tsit5(),
                                saveat=dfx.SaveAt(
                                    ts=np.linspace(t0, t1, save_steps)),
                                max_steps=max_steps,
                                stepsize_controller=make_stepsize_controller(
                                    t0=t0, t1=t1, dt0=dt0, dt1=dt1, choice=stepsize_controller),
                                # stepsize_controller=make_piecewise_stepcontrol(
                                #     t0=t0, t1=t1, dt0=dt0, dt1=dt1)
                                )))

    y00s, ts0 = simulate_steady_states(y0=y00, total_time=t1-t0, sim_func=sim_func, t0=t0,
                                       t1=t1, threshold=threshold, reverse_rates=reverse_rates, disable_logging=True)
    y0 = np.array(y00s[:, -1, :]).reshape(y00.shape)
    print('Steady states found. Now calculating signal response')

    # Signal
    y0m = y0 * ((signal_onehot == 0) * 1) + y00 * signal_target * signal_onehot
    ys, ts = simulate_steady_states(y0m, total_time=t1-t0, sim_func=sim_func, t0=t0,
                                    t1=t1, threshold=threshold, reverse_rates=reverse_rates, disable_logging=True)
    ys = np.concatenate([y0m, ys.squeeze()[:, :-1, :]], axis=1)

    analytics = jax.vmap(partial(compute_analytics, t=ts, labels=np.arange(
        ys.shape[-1]), signal_onehot=signal_onehot))(ys)

    return analytics, ys, ts, y0m, y00s, ts0


def setup_model(fake_circuits_reshaped, config_bio: dict, input_species: List[str]):
    model_brn = construct_model_fromnames(
        sample_names=input_species, include_prod_deg=config_bio['include_prod_deg'])
    ordered_species = sorted(get_unique(flatten_listlike(
        [r.input for r in model_brn.reactions if r.output])))
    for i in range(len(model_brn.reactions)):
        model_brn.reactions[i].forward_rate = 0
        model_brn.reactions[i].reverse_rate = 0
        if (not model_brn.reactions[i].input) and config_bio['include_prod_deg']:
            model_brn.reactions[i].forward_rate = config_bio['molecular_params'].get(
                'creation_rate')
        elif (not model_brn.reactions[i].output) and config_bio['include_prod_deg']:
            model_brn.reactions[i].forward_rate = config_bio['molecular_params'].get(
                'degradation_rate')

    qreactions = QuantifiedReactions()
    qreactions.init_properties(
        model_brn, config_bio.get('starting_concentration'))

    quantities = np.array(
        [r.quantity for r in qreactions.reactants if r.species.name in input_species])
    postproc = RawSimulationHandling(
        config_bio['interaction_simulator']).get_postprocessing(initial=quantities)

    # Update qreactions (using dummy rates)
    eqconstants, (a_rates, d_rates) = postproc(
        np.array(fake_circuits_reshaped[0]))
    model_brn, qreactions = update_species_simulated_rates(
        ordered_species, a_rates, d_rates, model_brn, qreactions)

    return model_brn, qreactions, ordered_species, postproc


def save(data_writer, analytics, ys, ts, y0m, fake_circuits, sampled_cond):
    print(data_writer.top_write_dir)
    data_writer.output(data=analytics, out_type='json', out_name='analytics')
    data_writer.output(data=ys, out_type='npy', out_name='ys')
    data_writer.output(data=ts, out_type='npy', out_name='ts')
    data_writer.output(data=y0m, out_type='npy', out_name='y0m')
    data_writer.output(data=fake_circuits, out_type='npy',
                       out_name='fake_circuits')
    data_writer.output(data=sampled_cond, out_type='npy',
                       out_name='sampled_cond')


def prep_sim(signal_species, qreactions, fake_circuits_reshaped, config_bio,
             forward_rates, reverse_rates):

    signal_onehot = np.where(
        [r.species.name in signal_species for r in qreactions.reactants], 1, 0)

    def make_flat_triangle(matrices):
        return np.array(list(map(lambda i: i[np.triu_indices(n=matrices.shape[-1])], matrices)))
    forward_rates, reverse_rates = make_flat_triangle(
        forward_rates), make_flat_triangle(reverse_rates)
    signal_target = config_bio['signal']['function_kwargs']['target']
    y00 = np.repeat(np.array([r.quantity for r in qreactions.reactants])[
                    None, None, :], repeats=len(fake_circuits_reshaped), axis=0)
    t0, t1, dt0, dt1, stepsize_controller = config_bio['simulation']['t0'], config_bio['simulation'][
        't1'], config_bio['simulation']['dt0'], config_bio['simulation']['dt1'], config_bio['simulation']['stepsize_controller']
    save_steps, max_steps = 50, 16**5

    return signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, save_steps, max_steps, forward_rates, reverse_rates


def prep_cfg(config_bio, input_species):

    config_bio = prepare_config(expand_config(config=config_bio))
    if config_bio.get('circuit_generation', {}).get('species_count') is not None:
        assert len(input_species) == config_bio.get('circuit_generation', {}).get(
            'species_count'), f'Wrong number of input species {input_species}'
    config_bio.update(expand_model_config(config_bio, {}, input_species))
    return config_bio


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
            purpose=config_bio['experiment']['purpose'], out_location=top_write_dir)
    config_bio, data_writer = script_preamble(
        config_bio, data_writer=data_writer)

    fake_circuits, z, sampled_cond = sample_reconstructions(params, rng, decoder,
                                                            n_categories=config_norm_y.categorical_n_bins if config_norm_y.categorical_n_bins else 10,
                                                            n_to_sample=n_to_sample, hidden_size=config_model.hidden_size,
                                                            x_datanormaliser=x_datanormaliser, x_methods_preprocessing=x_methods_preprocessing,
                                                            objective_cols=config_dataset.objective_col,
                                                            use_binned_sampling=use_binned_sampling, use_onehot=config_norm_y.categorical_onehot,
                                                            cond_min=cond.min(), cond_max=cond.max(), impose_final_range=impose_final_range, n_objectives=len(config_dataset.objective_col))

    # input_species = df[df['sample_name'].notna()]['sample_name'].unique()
    config_bio = prep_cfg(config_bio, input_species)

    fake_circuits_reshaped = make_batch_symmetrical_matrices(
        fake_circuits.reshape(-1, fake_circuits.shape[-1]), side_length=len(input_species))

    model_brn, qreactions, ordered_species, postproc = setup_model(
        fake_circuits_reshaped, config_bio, input_species)

    forward_rates, reverse_rates = make_rates(
        config_dataset.x_type, fake_circuits_reshaped, postproc)

    (signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, save_steps, max_steps, forward_rates, reverse_rates) = prep_sim(
        signal_species, qreactions, fake_circuits_reshaped, config_bio, forward_rates, reverse_rates)

    analytics, ys, ts, y0m, y00s, ts0 = sim(y00, forward_rates[0], reverse_rates,
                                            qreactions,
                                            signal_onehot, signal_target,
                                            t0, t1, dt0, dt1,
                                            save_steps, max_steps,
                                            stepsize_controller)

    analytics['sensitivity_wrt_species-6'] = np.array(
        analytics['sensitivity_wrt_species-6'])
    analytics['precision_wrt_species-6'] = np.array(
        analytics['precision_wrt_species-6'])
    analytics['overshoot'] = np.array(analytics['overshoot'])

    if visualise:
        vis_sampled_histplot(analytics['sensitivity_wrt_species-6'], model_brn, output_species, category_array=sampled_cond.reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='layer', save_path=os.path.join(data_writer.top_write_dir, 'sens_layer.png'))
        vis_sampled_histplot(analytics['sensitivity_wrt_species-6'], model_brn, output_species, category_array=sampled_cond.reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Sensitivity of generated circuits', x_label=f'Log10 of sensitivity to signal {signal_species}', multiple='fill', save_path=os.path.join(data_writer.top_write_dir, 'sens_fill.png'))
        vis_sampled_histplot(calculate_adaptation(analytics['sensitivity_wrt_species-6'], analytics['precision_wrt_species-6']), model_brn, output_species, category_array=sampled_cond.reshape(np.prod(sampled_cond.shape[:-1]), -1),
                             title=f'Adaptation of generated circuits', x_label=f'Adaptation to signal {signal_species}', multiple='layer', save_path=os.path.join(data_writer.top_write_dir, 'adapt_layer.png'))
    save(data_writer, analytics, ys, ts, y0m, fake_circuits, sampled_cond)

    if return_relevant:
        return analytics, ys, ts, y0m, y00s, ts0, fake_circuits, reverse_rates, model_brn, qreactions, ordered_species, input_species, z, sampled_cond
