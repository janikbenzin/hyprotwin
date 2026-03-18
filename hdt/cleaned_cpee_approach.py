from typing import Any, Callable, Tuple

from pandas import Timedelta
from pandas._libs import NaTType

from hdt.run.configuration import *
from hdt.util import *

from functools import partial

TIME_FREQUENCY = "s"
TIME_IN_STATE_FEATURE = "time_in_state_feature"
TIME_IN_STATE_FEATURE_1 = "time_in_state_feature_1"


def _compute_next_control(next_value, scale_max, scale_min, upper, lower,
                          past_series_bundle, scaler, series, inner_prediction_horizon, fc_cfg, last_controls, tp_cfg,
                          **kwargs):
    next_val_rescaled = next_value * (scale_max - scale_min) + scale_min
    new_date = pd.date_range(start=series[-1].start_time(),
                             periods=1,
                             freq=SAMPLING_RATE)
    if next_val_rescaled >= upper:
        next_control = 0.0
    elif next_val_rescaled < lower:
        next_control = 1.0
    else:
        next_control = last_controls.last_value()
    cs = TimeSeries.from_dataframe(
        pd.DataFrame(
            {last_controls.components[0]: [next_control]},
            index=new_date,
        ),
        freq=SAMPLING_RATE
    )
    return cs.astype(np.float32)

def _compute_next_control_multi(next_value, scale_max, scale_min, upper, lower,
                          past_series_bundle, scaler, series, inner_prediction_horizon, fc_cfg, last_controls, tp_cfg,
                          **kwargs):
    next_val_rescaled = next_value * (scale_max - scale_min) + scale_min
    new_date = pd.date_range(start=series[-1].start_time(),
                             periods=1,
                             freq=SAMPLING_RATE)
    # simulated
    sim_mode_component = get_all_sim_mode_components(fc_cfg.exp)[0]
    if next_val_rescaled >= upper:
        next_control = 0.0
    elif next_val_rescaled < lower:
        next_control = 1.0
    else:
        next_control = last_controls[sim_mode_component].last_value()
    if next_val_rescaled >= tp_cfg.actual_boundaries[1]:
        next_actual_control = 0.0
    elif next_val_rescaled < tp_cfg.actual_boundaries[0]:
        next_actual_control = 1.0
    else:
        next_actual_control = last_controls[fc_cfg.mode_component[0]].last_value()

    cs = TimeSeries.from_dataframe(
        pd.DataFrame(
            {sim_mode_component: [next_control],
                   fc_cfg.mode_component[0]: [next_actual_control]},

            index=new_date,
        ),
        freq=SAMPLING_RATE
    )
    return cs.astype(np.float32)


def _compute_next_control_sim_multistep(next_value, scale_max, scale_min, upper, lower,
                                        past_series_bundle, scaler, series, inner_prediction_horizon, fc_cfg,
                                        last_controls, tp_cfg, **kwargs):
    sim_mode_component = "mode_heat"
    new_sims = sim_past_covariates(series[-inner_prediction_horizon:], lower, upper, [scale_min], [scale_max], fc_cfg,
                                   last_control=last_controls[
                                       sim_mode_component].last_value())
    df = pd.DataFrame(
        data=new_sims,
        columns=[sim_mode_component],
        index=pd.date_range(
            start=series[-inner_prediction_horizon:].start_time(),
            periods=len(new_sims),
            freq=SAMPLING_RATE
        )
    )

    new_sims_ts = TimeSeries.from_dataframe(df, value_cols=[sim_mode_component])
    new_time_in_states = add_persistence_ts(
        concatenate([past_series_bundle[sim_mode_component][-tp_cfg.input_size:], new_sims_ts]))[
        -inner_prediction_horizon:]
    new_time_scaled = scaler.transform(new_time_in_states)
    new_bundle = concatenate([new_sims_ts, new_time_scaled, new_time_in_states], axis=1)
    past_series_bundle = concatenate([past_series_bundle[inner_prediction_horizon:], new_bundle]).astype(np.float32)

    return past_series_bundle


def _compute_next_control_multistep(next_value, scale_max, scale_min, upper, lower,
                                    past_series_bundle, series, inner_prediction_horizon, fc_cfg, last_controls,
                                    **kwargs):
    sim_mode_component = "mode_heat"
    new_sims = sim_past_covariates(series[-inner_prediction_horizon:], lower, upper, [scale_min], [scale_max], fc_cfg,
                                   last_control=last_controls[
                                       sim_mode_component].last_value())
    df = pd.DataFrame(
        data=new_sims,
        columns=[sim_mode_component],
        index=pd.date_range(
            start=series[-inner_prediction_horizon:].start_time(),
            periods=len(new_sims),
            freq=SAMPLING_RATE
        )
    )

    new_sims_ts = TimeSeries.from_dataframe(df, value_cols=[sim_mode_component]).astype(np.float32)
    return new_sims_ts


def _compute_next_control_sim(next_value, scale_max, scale_min, upper, lower, past_series_bundle,
                              scaler, series, **kwargs):
    next_val_rescaled = next_value * (scale_max - scale_min) + scale_min
    new_date = pd.date_range(start=series[-1].start_time(),
                             periods=1,
                             freq=SAMPLING_RATE)
    if next_val_rescaled >= upper:
        time_in_state = 0
        new_bundle = TimeSeries.from_times_and_values(new_date,
                                                      np.array([[
                                                          0.0,
                                                          time_in_state,
                                                          time_in_state]]),
                                                      columns=[
                                                          'mode_heat',
                                                          'time_in_state_feature',
                                                          'time_in_state_feature_1'])

    elif next_val_rescaled < lower:
        time_in_state = 0
        new_bundle = TimeSeries.from_times_and_values(new_date,
                                                      np.array([[
                                                          1.0,
                                                          time_in_state,
                                                          time_in_state]]),
                                                      columns=[
                                                          'mode_heat',
                                                          'time_in_state_feature',
                                                          'time_in_state_feature_1'])
    else:
        time_in_state = past_series_bundle["time_in_state_feature_1"].last_value() + 1
        unscaled_time_in_state = TimeSeries.from_times_and_values(new_date,
                                                                  np.array([[
                                                                      time_in_state]]),
                                                                  columns=[
                                                                      'time_in_state_feature_1'])
        time_in_state_scaled = scaler.transform(unscaled_time_in_state)
        new_bundle = TimeSeries.from_times_and_values(new_date,
                                                      np.array([[
                                                          past_series_bundle['mode_heat'].last_value(),
                                                          time_in_state_scaled.last_value(),
                                                          time_in_state]]),
                                                      columns=[
                                                          'mode_heat',
                                                          'time_in_state_feature',
                                                          'time_in_state_feature_1'])
    past_series_bundle = concatenate([past_series_bundle[1:], new_bundle]).astype(np.float32)

    return past_series_bundle


next_control_funcs = {
    "direct": _compute_next_control,
    "sim_scaled": _compute_next_control_sim,
    "sim_scaled_multistep": _compute_next_control_sim_multistep,
    "sim_multistep": _compute_next_control_multistep,
    "sim_multi": _compute_next_control_multi
}


def predict_simulative_control(chunks_test,
                               start_series,
                               fc_cfg,
                               tp_cfg,
                               eval_cfg,
                               model,
                               past_series,
                               upper,
                               lower,
                               base_time,
                               last_controls_init=lambda past_series, base_time, input_delta, mode_component:
                               past_series[base_time - input_delta:base_time][mode_component],
                               verbose=False,
                               multi=False,
                               simulated_covariates=None,
                               hmm_model=None,
                               inner_prediction_horizon=None
                               ):
    input_size = tp_cfg.input_size
    step_delta = pd.Timedelta(SAMPLING_FACTOR, "s")
    futures_all = []
    exp = fc_cfg.exp
    next_time = base_time
    mode_component = get_all_mode_names(exp)[0]
    prediction_horizon = tp_cfg.prediction_horizon
    print(f"Current prediction at {lower} {upper}", flush=True)
    input_delta = pd.Timedelta((input_size - 1) * SAMPLING_FACTOR, "s")
    scale_max = eval_cfg.scale_max[0]
    scale_min = eval_cfg.scale_min[0]
    model_type = fc_cfg.model_type
    if inner_prediction_horizon is None:
        if (not tp_cfg.sim):
            past_series_bundle = None
            # last_controls = past_series[base_time - input_delta:base_time][mode_component]
            last_controls = last_controls_init(past_series, base_time, input_delta, mode_component).astype(np.float32)
            compute_next_control = partial(next_control_funcs["direct"], scaler=None)
        else:
            p_simulated_covariates, all_past_covariates_sim = simulated_covariates
            last_controls = p_simulated_covariates[base_time - input_delta:base_time].astype(np.float32)
        if (tp_cfg.sim and hmm_model is None):
            past_series_bundle = None
            if multi:
                compute_next_control = partial(next_control_funcs["sim_multi"], scaler=None)
            else:
                compute_next_control = partial(next_control_funcs["direct"], scaler=None)
        elif (tp_cfg.sim and hmm_model is not None):
            past_series_bundle = all_past_covariates_sim[base_time - input_delta: base_time].astype(np.float32)
            compute_next_control = partial(next_control_funcs["sim_scaled"], scaler=hmm_model)
    else:
        num_threads = {}
        if hmm_model is None:
            p_simulated_covariates, all_past_covariates_sim = simulated_covariates
            last_controls = p_simulated_covariates[base_time - input_delta:base_time].astype(np.float32)
            past_series_bundle = None
            compute_next_control = partial(next_control_funcs["sim_multistep"], scaler=None)
        else:
            p_simulated_covariates, all_past_covariates_sim = simulated_covariates
            past_series_bundle = all_past_covariates_sim[base_time - input_delta: base_time].astype(np.float32)
            last_controls = p_simulated_covariates[base_time - input_delta:base_time].astype(np.float32)
            compute_next_control = partial(next_control_funcs["sim_scaled_multistep"], scaler=hmm_model)
    if verbose:
        last_controls_list = [last_controls]
    else:
        last_controls_list = None
    chunk_len = len(chunks_test)
    for chunk, chunk_data in enumerate(chunks_test):
        #ts_start = datetime.datetime.now()
        if chunk == 0:
            for i in range(prediction_horizon) if inner_prediction_horizon is None else range(0, prediction_horizon,
                                                                                              inner_prediction_horizon):
                if i > 0:
                    futures = futures.append(
                        model.predict(n=1 if inner_prediction_horizon is None else inner_prediction_horizon,
                                      series=series[-input_size:],
                                      past_covariates=last_controls[-input_size:],
                                      verbose=False))
                else:
                    series = get_sensors_only(start_series[-input_size:], fc_cfg).astype(np.float32)
                    futures = model.predict(n=1 if inner_prediction_horizon is None else inner_prediction_horizon,
                                            series=series[-input_size:], past_covariates=last_controls[-input_size:],verbose=False)

                series = series.append(futures[-1 if inner_prediction_horizon is None else -inner_prediction_horizon:])
                last_controls, next_time, past_series_bundle = compute_and_prepare_next_control(chunk,
                                                                                                compute_next_control,
                                                                                                futures,
                                                                                                last_controls,
                                                                                                last_controls_list,
                                                                                                lower,
                                                                                                mode_component,
                                                                                                next_time,
                                                                                                scale_max,
                                                                                                scale_min,
                                                                                                step_delta,
                                                                                                upper,
                                                                                                verbose,
                                                                                                past_series_bundle,
                                                                                                series[-input_size:],
                                                                                                i=i,
                                                                                                inner_prediction_horizon=inner_prediction_horizon,
                                                                                                fc_cfg=fc_cfg,
                                                                                                tp_cfg=tp_cfg)
        else:
            # have to reset to the correct end time
            next_time = chunk_data.start_time() - step_delta
            base_time = chunk_data.start_time() - step_delta
            if inner_prediction_horizon is None:
                if not tp_cfg.sim:
                    # last_controls = past_series[base_time - input_delta:base_time][mode_component]
                    last_controls = last_controls_init(past_series, base_time, input_delta, mode_component).astype(np.float32)
                #elif tp_cfg.sim and hmm_model is None:
                #    last_controls = last_controls_init(p_simulated_covariates, base_time, input_delta, mode_component)
                else:
                    last_controls = p_simulated_covariates[base_time - input_delta:base_time].astype(np.float32)
                if tp_cfg.sim and hmm_model is not None:
                    past_series_bundle = all_past_covariates_sim[base_time - input_delta: base_time].astype(np.float32)
            else:
                past_series_bundle = all_past_covariates_sim[base_time - input_delta: base_time].astype(np.float32)
                last_controls = p_simulated_covariates[base_time - input_delta:base_time].astype(np.float32)
            if verbose:
                last_controls_list.append(last_controls)
            for i in range(prediction_horizon):
                if i > 0:
                    futures = futures.append(
                        model.predict(n=1 if inner_prediction_horizon is None else inner_prediction_horizon,
                                      series=series[-input_size:],
                                      past_covariates=last_controls[-input_size:],
                                      verbose=False)
                    )

                else:
                    series = get_sensors_only(past_series[next_time - input_delta: next_time], fc_cfg).astype(np.float32)
                    futures = model.predict(n=1 if inner_prediction_horizon is None else inner_prediction_horizon,
                                            series=series[-input_size:],
                                            past_covariates=last_controls[-input_size:],
                                            verbose=False
                                            )


                series = series.append(
                    futures[-1 if inner_prediction_horizon is None else -inner_prediction_horizon:])
                last_controls, next_time, past_series_bundle = compute_and_prepare_next_control(chunk,
                                                                                                compute_next_control,
                                                                                                futures,
                                                                                                last_controls,
                                                                                                last_controls_list,
                                                                                                lower,
                                                                                                mode_component,
                                                                                                next_time,
                                                                                                scale_max,
                                                                                                scale_min,
                                                                                                step_delta,
                                                                                                upper,
                                                                                                verbose,
                                                                                                past_series_bundle,
                                                                                                series[
                                                                                                    -input_size:],
                                                                                                i,
                                                                                                inner_prediction_horizon,
                                                                                                fc_cfg=fc_cfg,
                                                                                                tp_cfg=tp_cfg)

        futures_all.append(futures)
        #ts_end = datetime.datetime.now()
        #print(f"Runtime: {ts_end - ts_start}")
        print(f"Chunk {chunk}/{chunk_len} finished", flush=True)
    if not verbose:
        return futures_all
    else:
        return futures_all, last_controls_list


def compute_and_prepare_next_control(chunk: int,
                                     compute_next_control: Callable[..., float | Any] | Callable[..., float | Any],
                                     futures: TimeSeries | Any,
                                     last_controls: TimeSeries | Any, last_controls_list: list[Any], lower,
                                     mode_component: str, next_time: Timedelta | NaTType | Any, scale_max, scale_min,
                                     step_delta: Timedelta | NaTType, upper, verbose: bool,
                                     past_series_bundle: Tuple[TimeSeries],
                                     series=None,
                                     i=None,
                                     inner_prediction_horizon=None,
                                     fc_cfg=None,
                                     tp_cfg=None) -> tuple[TimeSeries | Any, Timedelta | NaTType | Any, Any]:
    next_time += step_delta
    next_value = futures.last_value() if inner_prediction_horizon is None else futures[-inner_prediction_horizon:]
    past_series_bundle = compute_next_control(
        next_value,
        scale_max,
        scale_min,
        upper,
        lower,
        past_series_bundle=past_series_bundle,
        series=series,
        chunk=chunk,
        i=i,
        inner_prediction_horizon=inner_prediction_horizon,
        fc_cfg=fc_cfg,
        last_controls=last_controls,
        mode_component=mode_component,
        tp_cfg=tp_cfg
    )
    if TIME_IN_STATE_FEATURE in last_controls.components:
        if inner_prediction_horizon is None:
            cs = past_series_bundle[-1][[fc_cfg.mode_component[0], TIME_IN_STATE_FEATURE]]
        else:
            cs = past_series_bundle[-inner_prediction_horizon:][[fc_cfg.mode_component[0], TIME_IN_STATE_FEATURE]]
    else:
        cs = past_series_bundle
    last_controls = concatenate([last_controls, cs])
    if verbose:
        last_controls_list[chunk] = concatenate([last_controls_list[chunk], cs])
    return last_controls, next_time, past_series_bundle
