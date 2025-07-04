

from datetime import datetime
from typing import List
from evoscaper.utils.math import make_flat_triangle
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.srv.parameter_prediction.simulator import RawSimulationHandling
from synbio_morpher.utils.common.setup import prepare_config, expand_config, expand_model_config
from synbio_morpher.utils.misc.type_handling import flatten_listlike, get_unique
from synbio_morpher.utils.modelling.deterministic import bioreaction_sim_dfx_expanded
from synbio_morpher.utils.modelling.physical import eqconstant_to_rates
from synbio_morpher.utils.modelling.solvers import make_stepsize_controller
from synbio_morpher.utils.results.analytics.timeseries import generate_analytics
# from synbio_morpher.utils.modelling.solvers import simulate_steady_states, make_stepsize_controller
from bioreaction.model.data_tools import construct_model_fromnames
from bioreaction.model.data_containers import BasicModel, QuantifiedReactions
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx


def calc_num_unsteadied(comparison: np.ndarray, threshold: float):
    return np.nansum(np.abs(comparison) > threshold)


def did_sim_break(y):
    if (np.sum(np.isnan(y)) > 0):
        raise ValueError(
            f'Simulation failed - some runs ({np.sum(np.isnan(y))/np.size(y) * 100} %) go to nan. Try lowering dt.')
    if (np.sum(y == np.inf) > 0):
        raise ValueError(
            f'Simulation failed - some runs ({np.sum(y == np.inf)/np.size(y) * 100} %) go to inf. Try lowering dt.')


# @eqx.jit
# @jax.jit
def simulate_steady_states(y0, total_time, sim_func, t0, t1,
                           threshold=0.1, disable_logging=False,
                           **sim_kwargs):
    """ Simulate a function sim_func for a chunk of time in steps of t1 - t0, starting at 
    t0 and running until either the steady states have been reached (specified via threshold) 
    or until the total_time as has been reached. Assumes batching.

    Args:
    y0: initial state, shape = (batch, time, vars)
    t0: initial time
    t1: simulation chunk end time
    total_time: total time to run the simulation function over
    sim_kwargs: any (batchable) arguments left to give the simulation function,
        for example rates or other parameters. First arg must be y0
    threshold: minimum difference between the final states of two consecutive runs 
        for the state to be considered steady
    """

    ti = t0
    iter_time = datetime.now()
    # ys = y0
    # ys_full = ys
    # ts_full = 0
    while True:
        if ti == t0:
            y00 = y0
        else:
            y00 = ys[:, -1, :]

        ts, ys = sim_func(y00, **sim_kwargs)

        if np.sum(np.argmax(ts >= np.inf)) > 0:
            ys = ys[:, :np.argmax(ts >= np.inf), :]
            ts = ts[:, :np.argmax(ts >= np.inf)] + ti
        else:
            ys = ys
            ts = ts + ti

        did_sim_break(ys)

        if ti == t0:
            ys_full = ys
            ts_full = ts
        else:
            ys_full = np.concatenate([ys_full, ys], axis=1)
            ts_full = np.concatenate([ts_full, ts], axis=1)

        ti += t1 - t0

        if ys.shape[1] > 1:
            fderiv = jnp.gradient(ys[:, -3:, :], axis=1)[:, -1, :]  # / y00
        else:
            fderiv = ys[:, -1, :] - y00
        n_unsteadied = calc_num_unsteadied(fderiv / ys[:, -1, :], threshold)
        if (n_unsteadied == 0) or (ti >= total_time):
            if not disable_logging:
                print('Done: ', datetime.now() - iter_time)
            break
        if not disable_logging:
            print('Steady states: ', ti, ' iterations. ', n_unsteadied, ' left to steady out. ', datetime.now() - iter_time)

    if ts_full.ndim > 1:
        ts_full = ts_full[0]
    return np.array(ys_full), np.array(ts_full)
#


def compute_analytics(y, t, labels, signal_onehot):
    y = np.swapaxes(y, 0, 1)

    analytics_func = partial(
        generate_analytics, time=t, labels=labels,
        signal_onehot=signal_onehot, signal_time=t[1],
        ref_circuit_data=None)
    return analytics_func(data=y, time=t, labels=labels)


def make_rates(x_type, fake_circuits_reshaped, postprocs):
    if x_type == 'energies':
        eqconstants, (forward_rates, reverse_rates) = postprocs[x_type](
            fake_circuits_reshaped)
    elif x_type == 'eqconstants':
        forward_rates, reverse_rates = postprocs[x_type](
            fake_circuits_reshaped)
    elif x_type == 'binding_rates_dissociation':
        # reverse_rates = fake_circuits_reshaped
        # eqconstants = forward_rates[0, 0, 0] / reverse_rates
        forward_rates, reverse_rates = postprocs[x_type](
            fake_circuits_reshaped)
    else:
        raise ValueError(f'Unknown x_type {x_type}')
    return forward_rates, reverse_rates


def prep_sim_noconfig(signal_species, qreactions, fake_circuits_reshaped,
                      forward_rates, reverse_rates, signal_target, t0, t1, dt0, dt1, stepsize_controller):
    config_bio = {'signal': {'function_kwargs': {'target': signal_target}},
                  'simulation': {'t0': t0, 't1': t1, 'dt0': dt0, 'dt1': dt1, 'stepsize_controller': stepsize_controller}}
    return prep_sim(signal_species, qreactions, fake_circuits_reshaped, config_bio,
                    forward_rates, reverse_rates)


def prep_sim(signal_species: List[str], qreactions: QuantifiedReactions, fake_circuits_reshaped: np.ndarray, config_bio: dict,
             forward_rates: np.ndarray, reverse_rates: np.ndarray):

    signal_onehot = np.where(
        [r.species.name in signal_species for r in qreactions.reactants], 1, 0)

    forward_rates, reverse_rates = jax.vmap(make_flat_triangle)(
        forward_rates), jax.vmap(make_flat_triangle)(reverse_rates)
    signal_target = config_bio['signal']['function_kwargs']['target']
    y00 = np.repeat(np.array([r.quantity for r in qreactions.reactants])[
                    None, None, :], repeats=len(fake_circuits_reshaped), axis=0)

    # Get simulation settings
    simulation_settings = config_bio['simulation']
    t0 = simulation_settings['t0']
    t1 = simulation_settings['t1']
    dt0 = simulation_settings['dt0']
    dt1 = simulation_settings['dt1']
    tmax = simulation_settings.get(
        'tmax', simulation_settings.get('total_time', 10000))
    stepsize_controller = simulation_settings['stepsize_controller']
    threshold_steady_states = simulation_settings.get(
        'threshold_steady_states', 0.01)
    save_steps = simulation_settings.get('save_steps', 50)
    save_steps_uselog = simulation_settings.get('save_steps_uselog', False)
    if save_steps_uselog:
        save_steps = np.logspace(np.log10(t0 + 1), np.log10(t1), save_steps, base=10, endpoint=True) - 1.
        # save_steps[-1] = t1
    max_steps = simulation_settings.get('max_steps', (16**5) * 5)

    return signal_onehot, signal_target, y00, t0, t1, dt0, dt1, stepsize_controller, tmax, threshold_steady_states, save_steps, max_steps, forward_rates, reverse_rates


def prep_cfg(config_bio, input_species):
    config_bio = prepare_config(config_bio)
    if config_bio.get('circuit_generation', {}).get('species_count') is not None:
        assert len(input_species) == config_bio.get('circuit_generation', {}).get(
            'species_count'), f'Wrong number of input species {input_species}'
    config_bio.update(expand_model_config(config_bio, {}, input_species))
    return config_bio


def load_config_bio(fn, input_species, fn_simulation_settings=None):
    config_bio = load_json_as_dict(fn)
    config_bio_u = config_bio['base_configs_ensemble']['generate_species_templates']
    config_bio_u.update(
        config_bio['base_configs_ensemble']['mutation_effect_on_interactions_signal'])
    config_bio = prep_cfg(config_bio_u, input_species)
    if fn_simulation_settings is not None:
        config_sim = load_json_as_dict(fn_simulation_settings)
        config_bio['simulation'].update(config_sim['simulation'])
        config_bio['simulation_steady_state'].update(
            config_sim['simulation_steady_state'])
    return config_bio


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


def setup_model_brn(config_bio: dict, input_species: List[str]):

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
    k_a = RawSimulationHandling(
        config_bio['interaction_simulator']).fixed_rate_k_a

    postprocs = {
        'energies': RawSimulationHandling(config_bio['interaction_simulator']).get_postprocessing(initial=quantities),
        'eqconstants': partial(eqconstant_to_rates, k_a=k_a),
        'binding_rates_dissociation': lambda x: (np.ones_like(x) * k_a, x)
    }

    return model_brn, qreactions, postprocs, ordered_species


def setup_model(fake_circuits_reshaped, config_bio: dict, input_species: List[str], x_type: str):

    model_brn, qreactions, postprocs, ordered_species = setup_model_brn(
        config_bio, input_species)

    # Update qreactions (using dummy rates)
    forward_rates, reverse_rates = make_rates(
        x_type, fake_circuits_reshaped, postprocs)

    model_brn, qreactions = update_species_simulated_rates(
        ordered_species, forward_rates, reverse_rates, model_brn, qreactions)

    return model_brn, qreactions, ordered_species, postprocs


def reduce_tdim(ts, ys):
    if ts.ndim > 1:
        if ys.shape[0] == ts.shape[0]:
            ts = ts[0]
        if ys.shape[-1] == ts.shape[-1]:
            ts = ts[..., 0]
    return ts


def sim_core(
        y00, forward_rates, reverse_rates,
        qreactions,
        signal_onehot, signal_target,
        t0, t1, dt0, dt1,
        save_steps, max_steps,
        stepsize_controller,
        dt1_factor=5,
        threshold=0.01,
        total_time=None,
        disable_logging=False):

    rate_max = np.max([np.max(forward_rates),
                       np.max(reverse_rates)])
    dt0 = np.min([1 / (5 * rate_max), dt0])
    dt1 = dt1_factor * dt0
    total_time = t1 - t0 if total_time is None else total_time
    save_steps_ts = np.linspace(t0, t1, save_steps) if type(save_steps) == int else save_steps

    sim_func = jax.jit(jax.vmap(partial(bioreaction_sim_dfx_expanded,
                                t0=t0, t1=t1, dt0=dt0,
                                signal=None, signal_onehot=None,
                                inputs=qreactions.reactions.inputs,
                                outputs=qreactions.reactions.outputs,
                                forward_rates=forward_rates,
                                solver=dfx.Tsit5(),
                                saveat=dfx.SaveAt(
                                    ts=save_steps_ts),
                                max_steps=max_steps,
                                stepsize_controller=make_stepsize_controller(
                                    t0=t0, t1=t1, dt0=dt0, dt1=dt1, choice=stepsize_controller),
                                # stepsize_controller=make_piecewise_stepcontrol(
                                #     t0=t0, t1=t1, dt0=dt0, dt1=dt1)
                                )))

    print(
        f'Simulating steady states for {y00.shape[0]} circuits for a max of {total_time} time units')
    time_start = datetime.now()
    y00s, ts0 = simulate_steady_states(y0=y00, total_time=total_time, sim_func=sim_func, t0=t0,
                                       t1=t1, threshold=threshold, reverse_rates=reverse_rates, disable_logging=disable_logging)
    ts0 = reduce_tdim(ts0, y00s)
    y0 = np.array(y00s[:, -1, :]).reshape(y00.shape)
    minutes, seconds = divmod(
        (datetime.now() - time_start).total_seconds(), 60)
    print(
        f'Steady states found after {int(minutes)} mins and {int(seconds)} secs. Now calculating signal response')

    # Signal
    # y0m = y0 * ((signal_onehot == 0) * 1) + y00 * signal_target * signal_onehot
    y0m = y0 * (signal_target * signal_onehot + (signal_onehot == 0) * 1)
    time_start = datetime.now()
    ys, ts = simulate_steady_states(y0m, total_time=total_time, sim_func=sim_func, t0=t0,
                                    t1=t1, threshold=threshold, reverse_rates=reverse_rates, disable_logging=disable_logging)
    ts = reduce_tdim(ts, ys)
    minutes, seconds = divmod(
        (datetime.now() - time_start).total_seconds(), 60)
    print(
        f'Signal response found after {int(minutes)} mins and {int(seconds)} secs.')
    ys = np.concatenate([y0m, ys.squeeze()[:, :-1, :]], axis=1)

    return ys, ts, y0m, y00s, ts0


def sim(y00, forward_rates, reverse_rates,
        qreactions,
        signal_onehot, signal_target,
        t0, t1, dt0, dt1,
        save_steps, max_steps,
        stepsize_controller,
        dt1_factor=5,
        threshold=0.01,
        total_time=None,
        disable_logging=False):
    """ Concentrations should be in the form [circuits, time, species] """

    ys, ts, y0m, y00s, ts0 = sim_core(y00, forward_rates, reverse_rates,
                                      qreactions, signal_onehot, signal_target,
                                      t0, t1, dt0, dt1, save_steps, max_steps,
                                      stepsize_controller, dt1_factor, threshold, total_time, disable_logging)

    analytics = jax.vmap(partial(compute_analytics, t=ts, labels=np.arange(
        ys.shape[-1]), signal_onehot=signal_onehot))(ys)

    return analytics, ys, ts, y0m, y00s, ts0
