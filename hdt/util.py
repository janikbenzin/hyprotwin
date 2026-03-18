import os
import pickle
from dataclasses import replace
from typing import Any, Literal

import darts.metrics
import matplotlib.pyplot as plt
import json

import pandas
import pandas as pd
import numpy as np
import datetime
import re
import onnx
import base64
import torch
import traceback

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar

import packaging.version as pv
import warnings
from lightgbm import LGBMRegressor
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.shape_calculator import (
    calculate_linear_regressor_output_shapes,
)

from onnxmltools import __version__ as oml_version
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
    convert_lightgbm,
)

from hdt.forecasting_models.constructor import get_constructor
from hdt.parameters import *
from hdt.run.configuration import WorkerBundle, TrainPredictConfig


def assign_intermediate(all_intermediate, exp, model, st_lt, segmentation, component, result_value, prediction_horizon, prediction_stride, input_size=None, result_key=None, score_key=None, mode=None, exist=None, reset=None):
    try:
        if score_key is not None:
            if mode is None:
                if exist is not None and reset is None:
                    if exist is True:
                        return all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][str(prediction_horizon)][str(input_size)][str(prediction_stride)][scores][score_key] is not None
                    else:
                        return all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][
                            str(prediction_horizon)][str(input_size)][str(prediction_stride)][
                            scores][score_key]
                elif reset is not None:
                    for sk in all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][str(prediction_horizon)][str(input_size)][str(prediction_stride)][scores]:
                        all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][str(prediction_horizon)][str(input_size)][str(prediction_stride)][scores][sk] = None
                    return
                else:
                    all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][str(prediction_horizon)][str(input_size)][str(prediction_stride)][scores][score_key] = result_value
                    #print(f"Evaluation of {model.value} for {exp} in setting {st_lt.value} horizon {prediction_horizon} input size {input_size} and {segmentation} with components {component} shows for {score_key} a value of: {result_value}.\n")
            else:
                all_intermediate[exp][model.value][segmentation.value][component.value][str(mode)][str(prediction_horizon)][str(input_size)][str(prediction_stride)][scores][
                    score_key] = result_value
        else:
            if mode is None:
                all_intermediate[exp][model.value][segmentation.value][component.value][st_lt.value][str(prediction_horizon)][str(input_size)][str(prediction_stride)][result_key] = result_value
            else:
                all_intermediate[exp][model.value][segmentation.value][component.value][str(mode)][str(prediction_horizon)][str(input_size)][str(prediction_stride)][result_key] = result_value
    except KeyError as e:
        import traceback
        traceback.print_exc()
        print(f"KeyError for {(all_intermediate, exp, model, st_lt, segmentation, component, result_value, prediction_horizon, prediction_stride)}")

def generate_torch_kwargs():
    # run torch forecasting_models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [TFMProgressBar(enable_train_bar_only=True)],
        }
    }


def encode_modes(exp, cell, overwrite=False):
    if exp in [HEATING, HEATING_2H]:
        if cell == "On":
            return 1
        else:
            return 0
    elif exp == WATERLEVEL:
        # Waterlevel has a reversed On/Off logic for the simulated control modes
        if cell == "On":
            return 0
        else:
            return 1
    elif exp == IRRIGATION:
        if not overwrite:
            if cell == "Irrigation On":
                return 1
            else:
                return 0
        else:
            if cell == "On":
                return 1
            else:
                return 0
    else:
        return cell


def decode_modes(exp, mode):
    if exp == HEATING or exp == HEATING_2H:
        if mode == 0:
            return "false"
        else:
            return "true"
    else:
        return mode


# TODO change for multivariate time series (per component figure)
def display_forecast(pred_series, ts_transformed, forecast_type, figure_path=None, start_date=None):
    plt.figure(figsize=(12, 8))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.univariate_component(0).plot(label=("historic " + forecast_type + " forecasts"))
    from darts.metrics import r2_score
    plt.title(f"R2: {r2_score(ts_transformed.univariate_component(0), pred_series.univariate_component(0))}")
    plt.legend()
    plt.savefig(figure_path)
    plt.show()


def chunk_series(ts: TimeSeries, chunk_timedelta: pd.Timedelta, stride_timedelta: pd.Timedelta = None):
    if stride_timedelta is None:
        stride_timedelta = chunk_timedelta
    chunks = []
    start = ts.start_time()
    end = ts.end_time()
    cur = start
    while cur + chunk_timedelta <= end:
        #
        chunk = ts.slice(cur, cur + chunk_timedelta - ts.freq.delta)
        chunks.append(chunk)
        cur = cur + stride_timedelta
    return chunks


def get_data_path(exp):
    return f"{PATH_PREFIX}/data/{exp}/"

def get_bundles_path(exp, fold):
    return f"{get_data_path(exp)}fold{fold}.pkl"

def get_hdt_data_path(exp):
    return f"{PATH_PREFIX}/data/hdt_evaluation/{exp}/"


def get_models_path(exp, mt, segmentation, st_lt):
    return f"{PATH_PREFIX}/darts_logs/{mt.value}_{exp}/{segmentation.value}/{st_lt.value}/"


def get_constant_dataset(exp, scenario=''):
    return f"{get_data_path(exp)}{scenario}{exp}_constant_rate.csv"


def get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, serialization_t="pkl", mode=None):
    if serialization_t == "pkl":
        return f"{get_models_path(exp, mt, segmentation, st_lt)}{get_model_name(exp, mt, component, prediction_horizon, input_size)}{'' if mode is None else "_" + str(mode)}.pkl"
    else:
        return f"{get_models_path(exp, mt, segmentation, st_lt)}{get_model_name(exp, mt, component, prediction_horizon, input_size)}{'' if mode is None else "_" + str(mode)}.onnx"


def scale_min_max(value, scale_min, scale_max):
    return (value - scale_min) /  (scale_max - scale_min)


def sim_past_covariates(series, lower, upper, scale_min, scale_max, fc_cfg, series2=None, pad=False, last_control=None):
    # Extract sensor data once using vectorized operations
    sensor_values = get_sensors_only(series, fc_cfg).values().flatten()
    if last_control is None:
        mode_values = series[get_all_mode_names(fc_cfg.exp)].values().flatten()
        # Sequential computation (must be done in order due to state dependency)
        last_control = np.int32(mode_values[0])

    # Vectorized rescaling for all sensor values at once
    sensor_rescaled = sensor_values * (scale_max[0] - scale_min[0]) + scale_min[0]

    # Initialize result array
    past_covariates_sim = np.empty(len(series), dtype=np.int32)

    for i in range(len(series)):
        if i > 0:
            next_val_rescaled = sensor_rescaled[i]

            if next_val_rescaled >= upper:
                last_control = np.int32(0)
            elif next_val_rescaled < lower:
                last_control = np.int32(1)
            # else: keep last_control unchanged

        past_covariates_sim[i] = last_control

    if False:# Handle the final element
        if not pad and series2 is not None:
            series2_sensor = get_sensors_only(series2[0:1], fc_cfg).values()[0][0]
            series2_rescaled = series2_sensor * (scale_max[0] - scale_min[0]) + scale_min[0]

            if series2_rescaled >= upper:
                final_control = np.int32(0)
            elif series2_rescaled < lower:
                final_control = np.int32(1)
            else:
                final_control = last_control

            past_covariates_sim = np.append(past_covariates_sim, final_control)
        elif pad:
            past_covariates_sim = np.append(past_covariates_sim, last_control)

    return past_covariates_sim

def get_raw_datasets():
    return f"{PATH_PREFIX}/data/all_raw_timeseries.json"


def get_figure_path(exp, tmode, mt, st_lt, component, segmentation, stride, input_size, mode=None):
    return f"{PATH_PREFIX}/figures/{tmode}_{exp}_{mt.value}_{st_lt.value}_{component.value}_{segmentation.value}_{stride}_{input_size}{'' if mode is None else mode}.png"


def store_model(exp, model, mt, st_lt, component, segmentation, prediction_horizon, input_size, mode=None):
    model_path = get_models_path(exp, mt, segmentation, st_lt)
    if os.path.exists(model_path):
        model.save(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, mode=mode))
    else:
        os.makedirs(model_path)
        model.save(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, mode=mode))


def predict_future(model, bundle, tp_cfg, w_bundle, past_series):
    return model.historical_forecasts(
        past_series,
        start=bundle.test_scaled.start_time(),
        forecast_horizon=tp_cfg.prediction_horizon,
        stride=tp_cfg.prediction_stride,
        past_covariates=w_bundle.all_past_covariates,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )

def predict_future_slow(model, bundle, fc_cfg, tp_cfg, w_bundle, past_series):
    futures = []
    step_delta = pd.Timedelta(SAMPLING_FACTOR, "s")
    input_delta = pd.Timedelta((tp_cfg.input_size - 1) * SAMPLING_FACTOR, "s")
    base_time = bundle.val_scaled.end_time()
    val_scaled = get_sensors_only(bundle.val_scaled, fc_cfg)
    chunk_len = len(w_bundle.chunks)
    for chunk, chunk_data in enumerate(w_bundle.chunks):
        if chunk == 0:
            futures.append(model.predict(n=tp_cfg.prediction_horizon, series=val_scaled[-tp_cfg.input_size:],
                                         past_covariates=w_bundle.all_past_covariates if w_bundle.all_past_covariates is None else w_bundle.all_past_covariates[
                                             base_time - input_delta: base_time]))
        else:
            base_time = chunk_data.start_time() - step_delta
            futures.append(model.predict(n=tp_cfg.prediction_horizon, series=past_series[base_time - input_delta: base_time],
                                         past_covariates=w_bundle.all_past_covariates if w_bundle.all_past_covariates is None else w_bundle.all_past_covariates[
                                             base_time - input_delta: base_time]))
        print(f"Finished chunk {chunk}/{chunk_len}")
    return futures


def get_model_name(exp, mt, segmentation, prediction_horizon, input_size):
    return f"{mt.value}_{exp}_{segmentation.value}_{prediction_horizon}_{input_size}"


def convert_enum_keys(obj):
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): convert_enum_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [convert_enum_keys(i) for i in obj]
    return obj

def current_timestamp():
    return datetime.datetime.now()


def custom_json_serializer(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Obj {obj.__class__.__name__} is not JSON serializable")

def store_intermediate(current, eval=False):
    serialized_current = convert_enum_keys(current)
    if not eval:
        with open(f"{PATH_PREFIX}/tmp/train_predict_baseline_inf_{current_timestamp()}.json", "w") as f:
            json.dump(serialized_current, f, default=custom_json_serializer)
    else:
        with open(f"{PATH_PREFIX}/tmp/final_results_dict_{current_timestamp()}.json", "w") as f:
            json.dump(serialized_current, f, default=custom_json_serializer)


def inject_results_into_intermediate(exp):
    # Load the latest intermediate JSON
    all_intermediate = load_latest_intermediate()

    # If no existing intermediate found, initialize an empty dict
    if all_intermediate is None:
        all_intermediate = {}

    # Get the results dictionary for this experiment
    results = get_results_dict(exp)

    # Inject the results into the all_intermediate structure
    # Navigate through the nested structure and inject at the appropriate level
    if exp not in all_intermediate:
        all_intermediate[exp] = get_intermediate(exp)
    else:
        # Inject results into existing structure
        for model in Models:
            if model.value in all_intermediate[exp]:
                # For Segmentation OFF
                if Segmentation.OFF.value in all_intermediate[exp][model.value]:
                    for component in Components:
                        if component.value in all_intermediate[exp][model.value][Segmentation.OFF.value]:
                            for st_lt in Horizon:
                                if st_lt.value in all_intermediate[exp][model.value][Segmentation.OFF.value][
                                    component.value]:
                                    # Merge the results dict
                                    all_intermediate[exp][model.value][Segmentation.OFF.value][component.value][
                                        st_lt.value] = results

                # For Segmentation ON
                if Segmentation.ON.value in all_intermediate[exp][model.value]:
                    for component in Components:
                        if component.value in all_intermediate[exp][model.value][Segmentation.ON.value]:
                            for mode in modes_assignment[exp]:
                                if mode in all_intermediate[exp][model.value][Segmentation.ON.value][component.value]:
                                    # Merge the results dict
                                    all_intermediate[exp][model.value][Segmentation.ON.value][component.value][
                                        mode] = results

    # Store the updated intermediate with current timestamp
    store_intermediate(all_intermediate)

    print(f"Successfully injected results for experiment '{exp}' into all_intermediate")
    return all_intermediate


def overwrite_intermediate(exp):
    # Load the latest intermediate JSON
    all_intermediate = load_latest_intermediate()

    # If no existing intermediate found, initialize an empty dict
    if all_intermediate is None:
        all_intermediate = {}

    # Get the results dictionary for this experiment
    results = get_intermediate(exp)

    # Inject the results into the all_intermediate structure
    # Navigate through the nested structure and inject at the appropriate level
    all_intermediate[exp] = results
    # Store the updated intermediate with current timestamp
    store_intermediate(all_intermediate)

    print(f"Successfully injected results for experiment '{exp}' into all_intermediate")
    return all_intermediate



def load_latest_intermediate(path_prefix=f"{PATH_PREFIX}/tmp/", pattern=r"train_predict_baseline_inf_(.+)\.json"):

    latest_file = None
    latest_ts = None

    for filename in os.listdir(path_prefix):
        match = re.match(pattern, filename)
        if not match:
            continue

        ts_str = match.group(1)

        try:
            ts = datetime.datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_file = os.path.join(path_prefix, filename)

    if latest_file is None:
        return None

    # Load the file
    with open(latest_file, "r") as f:
        all_intermediate = json.load(f)
        print(f"Loaded latest intermediate JSON from {latest_ts}")
    return all_intermediate


def check_existing_model(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, serialization_t="pkl", mode=None):
    return os.path.exists(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, serialization_t, mode=mode))


def get_model_path(intermediate, fc_cfg, tp_cfg, mode):
    if mode is None:
        return intermediate[fc_cfg.exp][fc_cfg.model_type.value][tp_cfg.segmentation.value][tp_cfg.component.value][fc_cfg.st_lt.value][str(tp_cfg.prediction_horizon)][str(tp_cfg.input_size)][str(tp_cfg.prediction_stride)][actual_model]
    else:
        return intermediate[fc_cfg.exp][fc_cfg.model_type.value][tp_cfg.segmentation.value][tp_cfg.component.value][str(mode)][str(tp_cfg.prediction_horizon)][str(tp_cfg.input_size)][str(tp_cfg.prediction_stride)][actual_model]

def get_model(intermediate, model_class, fc_cfg, tp_cfg, mode=None):
    return model_class.load(get_model_path(intermediate, fc_cfg, tp_cfg, mode))

def store_onnx_model(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, model, mode=None):
    model_path = get_models_path(exp, mt, segmentation, st_lt)
    if os.path.exists(model_path):
        with open(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, "onnx", mode), "wb") as f:
            f.write(model.SerializeToString())
    else:
        os.makedirs(model_path)
        with open(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, "onnx", mode), "wb") as f:
            f.write(model.SerializeToString())


def load_onnx_model(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, mode=None):
    return onnx.load(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon,  input_size, "onnx", mode))

def segment_by_control_mode(series, control_column, value_columns, lags, pad=True):
    mode_series = series[control_column].values().ravel()
    modes = set(mode_series)
    segments = {mode: [] for mode in modes}
    start_i = 0
    start_mode = mode_series[0]
    for i in range(1, len(mode_series)):
        if mode_series[i] != mode_series[i - 1]:
            if start_i > 0 and pad:
                seg = series[start_i-lags:i][value_columns]
            else:
                seg = series[start_i:i][value_columns]
            if len(seg) > 0:
                segments[start_mode].append(seg)
            start_i = i
            start_mode = mode_series[i]
    return segments


def initialize_parameters_st_lt(parameters, st_lt):
    for key, item in parameters.items():
        if isinstance(item, dict):
            parameters[key] = parameters[key][st_lt]
    return parameters

def get_sensors_only(series, forecast_config):
    if isinstance(series, list):
        return [inner_series[get_sensor_components(inner_series, forecast_config)] for inner_series in series]
    else:
        return series[get_sensor_components(series, forecast_config)]



def get_sensor_components(series, fc_cfg):
    return [var for var in series.components.to_list() if var.startswith(fc_cfg.sensor_prefix)]
#def convert_model_to_onnx(model, exp, mt, input_size, n_components, output_horizon, st_lt, component, segmentation, mode):
#    if mt is Models.RF:
#        initial_type = [('input', FloatTensorType([None, input_size * n_components]))]
#        onnx_model = convert_sklearn(
#            model.get_estimator(input_size, output_horizon), initial_types=initial_type)
#        store_onnx_model(exp=exp, mt=mt, st_lt=st_lt, component=component, segmentation=segmentation, prediction_horizon=prediction_horizon,
#                         prediction_stride=prediction_stride, model=onnx_model, mode)
#    else:
#        pass


def convert_ts_to_flattened_onnx_input(ts, st_input_size):
    input_row_vector = []
    ts_components = ts.components.to_list()
    for i in [k for k, comp in enumerate(ts.components) if comp != MODE_PREFIX]:
        for j in range(st_input_size):
            input_row_vector.append(ts[-(st_input_size - j)][ts_components[i]].values()[0][0])
    return np.array([input_row_vector])


def load_and_runtime_onnx_model(exp, mt, st_lt,component, segmentation, prediction_horizon, input_size, mode):
    import onnxruntime as rt
    sess = rt.InferenceSession(get_simple_model_path(exp, mt, st_lt, component, segmentation, prediction_horizon, input_size, "onnx", mode),
                               providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def get_input_vector_path(exp, st_input_size):
    return f"{get_data_path('input_vectors')}{exp}_{st_input_size}_lags.json"


def store_input_vector_for_cpee(exp, val_scaled, st_input_size):
    with open(get_input_vector_path(exp, st_input_size), "w") as f:
        json.dump(val_scaled[-st_input_size:].to_json(), f)


def convert_to_onnx_input(series, past_covariates):
    stacked = series.stack(past_covariates)
    input_vector = stacked.values()
    flat = input_vector.flatten(order='F')
    return flat.reshape(1, -1).astype(np.float32)


def predict_onnx(sess, series, past_covariates, input_size):
    input_name = sess.get_inputs()[0].name
    stacked = series.stack(past_covariates)
    input_vector = stacked[-input_size:].values()
    flat = input_vector.flatten(order='F')
    input_reshaped = flat.reshape(1, -1).astype(np.float32)
    return sess.run(None, {input_name: input_reshaped})[0]


def convert_and_store_onnx_model(exp, model, model_type, st_lt, component, segmentation, prediction_horizon, input_size, mode, series=None, no_components=None):
    try:
        if model_type in [Models.LightGBM]:
            #if no_components is None:
            #    no_components = len(series.components)
            #X = np.random.uniform(low=0.0, high=1.0, size=(input_size,no_components))
            model_estimator = model.model
            onnx_model = to_onnx(
                model_estimator,
                #X.astype(np.float32),
                initial_types=[("X", FloatTensorType(shape=[None, len(model_estimator.feature_name_)]))],
                target_opset={"": 14, "ai.onnx.ml": 2},
                options={'split': 50}
                #options={"split": 100},
            )
            store_onnx_model(exp=exp, mt=model_type, st_lt=st_lt, component=component, segmentation=segmentation,
                             prediction_horizon=prediction_horizon, input_size=input_size, model=onnx_model, mode=mode)
            return True
        elif model_type is Models.DLinear:
            # DLinear from Darts is a TorchForecastingModel
            onnx_path = get_simple_model_path(exp, model_type, st_lt, component, segmentation, prediction_horizon,
                                              input_size, "onnx", mode)
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            model.to_onnx(path=onnx_path, export_params=True,
                opset_version=14)
            return True
        else:
            # Both LSTM and TCN
            onnx_path = get_simple_model_path(exp, model_type, st_lt, component, segmentation, prediction_horizon,
                                              input_size, "onnx", mode)
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            model.to_onnx(path=onnx_path, export_params=True,
                          opset_version=14)
            return True
    except Exception as e:
        print(
            f"ONNX serialization error occured for {exp} {model} {model_type} {st_lt} {component} {segmentation} {prediction_horizon} {input_size} {mode}. Continuing...")
        traceback.print_exc()
        return False



## Source: https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_lightgbm_reg.html
def skl2onnx_convert_lightgbm(scope, operator, container):
    options = scope.get_options(operator.raw_operator)
    if "split" in options:
        if pv.Version(oml_version) < pv.Version("1.9.2"):
            warnings.warn(
                "Option split was released in version 1.9.2 but %s is "
                "installed. It will be ignored." % oml_version,
                stacklevel=0,
            )
        operator.split = options["split"]
    else:
        operator.split = None
    convert_lightgbm(scope, operator, container)


update_registered_converter(
    LGBMRegressor,
    "LightGbmLGBMRegressor",
    calculate_linear_regressor_output_shapes,
    skl2onnx_convert_lightgbm,
    options={"split": None},
)

def get_print_string(exp, model_type, st_lt, component, segmentation, prediction_horizon, prediction_stride, input_size):
    return f"Experiment: {exp}, model: {model_type}, horizon: {st_lt}, component: {component}, segment: {segmentation}, prediction horizon: {prediction_horizon}, input_size: {input_size}, stride: {prediction_stride}"


def generate_superprocess_forecast_input_json(all_intermediate, bundle, cpee_predictive_simulation, forecast_config, all_modes=False):
    test_scaled = bundle.test_scaled
    val_scaled = bundle.val_scaled
    exp = forecast_config.exp
    superprocess_dict = {}
    step_delta = pd.Timedelta(SAMPLING_FACTOR, "s")
    prediction_horizons = RUN_PREDICTION_HORIZONS[exp]
    strides = RUN_PREDICTION_STRIDES[exp]
    chunks = None
    all_sensors = concatenate([get_sensors_only(val_scaled, forecast_config), get_sensors_only(test_scaled, forecast_config)])
    all_combs = {(comb[0], comb[1], comb[2]) for comb in cpee_predictive_simulation.keys()}
    best_bundles = {}
    for horizon in prediction_horizons:
        st_input_sizes = RUN_INPUT_SIZES[exp][horizon]
        print(st_input_sizes)
        superprocess_dict[horizon] = {}
        for stride in strides:
            superprocess_dict[horizon][stride] = {}
            for size in st_input_sizes:
                chunk_duration = pd.Timedelta(horizon * SAMPLING_FACTOR, "s")
                stride_duration = pd.Timedelta(stride * SAMPLING_FACTOR, "s")
                chunks = chunk_series(test_scaled, chunk_duration, stride_duration)
                superprocess_dict[horizon][stride][size] = {}
                comb = (stride, horizon, size)
                if comb in all_combs:
                    next_comb = next((t for t in cpee_predictive_simulation.keys() if t[:3] == comb), None)
                    tp_cfg, w_bundles = cpee_predictive_simulation[next_comb]
                    best, r = select_best_model_by_rmse(all_intermediate, tp_cfg, forecast_config)
                    seg = best['seg']
                    mode = best['mode']
                    if seg is Segmentation.OFF:
                        w_bundle = w_bundles[0]
                        w_bundle = replace(w_bundle, chunks=chunks)
                        best_bundles[next_comb] = w_bundle
                    elif mode == '0.0':
                        w_bundle = w_bundles[1]
                        w_bundle = replace(w_bundle, chunks=chunks)
                        best_bundles[next_comb] = w_bundle
                    else:
                        w_bundle = w_bundles[2]
                        w_bundle = replace(w_bundle, chunks=chunks)
                        best_bundles[next_comb] = w_bundle
                    #superprocess_dict[horizon][stride][size][SENSOR_PREFIX] = {}
                    #superprocess_dict[horizon][stride][size][MODE_PREFIX] = {}
                    for i in range(len(chunks)):
                        base_time = chunks[i].start_time() - step_delta
                        start_time = base_time - (size - 1) * step_delta
                            #control_modes = [decode_modes(exp, val_scaled[get_all_mode_names(exp)].values()[-1][0])]
                            #print(w_bundle.all_past_covariates[base_time - (size * step_delta): base_time].start_time(), w_bundle.all_past_covariates[base_time - (size * step_delta): base_time].end_time())
                            #print(chunks[i].start_time())
                        covariates = w_bundle.all_past_covariates[start_time:base_time]
                        sensors = all_sensors[start_time:base_time]
                        input_vector = convert_to_onnx_input(sensors, covariates)
                        superprocess_dict[horizon][stride][size][i] = {
                                forecast_config.sensor_prefix: input_vector,
                                forecast_config.mode_prefix: covariates.values().flatten().tolist() if all_modes else covariates[-1].values().astype(np.int32).flatten().tolist()}
                            #superprocess_dict[horizon][stride][size][MODE_PREFIX][i] = }

                            #control_modes = [decode_modes(exp, chunks[i-1][get_all_mode_names(exp)].values()[-1][0])]
    return superprocess_dict, best_bundles


def get_mode_name(exp, i):
    return MODE_PREFIX + MODES[exp][i]


def get_all_mode_names(exp):
    return [MODE_PREFIX + n for n in MODES[exp]]

def get_all_sim_mode_components(exp):
    return [n + "_sim" for n in get_all_mode_names(exp)]

def extract_sub_from_index(dir_path):
    with open(dir_path + '/index.txt') as f:
        indented_text = f.read()

    t = [line for line in indented_text.splitlines() if line.strip()]
    subprocesses = {ot.strip().split('(')[1].split(')')[0].strip(): ot.strip().split('(')[0].strip() for ot in
                    indented_text.splitlines() if ot.strip()}
    return subprocesses, t


def get_log_name(sid):
    return f"{sid}.xes.yaml"

def get_hdt_path(exp):
    return f"{get_data_path("hdt_evaluation")}{exp}/"


def quick_evaluation(chunks_test, futures_all, fc_cfg, tp_cfg):
    avgs = []
    for score in Scores:
        result = []
        for test, future in zip(chunks_test, futures_all):
            result.append(metric_assignment[score](get_sensors_only(test, fc_cfg), future))
        avgs.append(np.average(result))
        print(
            f"The average {score.value} is {avgs[-1]} for {get_print_string(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation, tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size)}")
    return avgs



def quick_accuracy(chunks_test, futures_all, fc_cfg, tp_cfg):
    result = []
    for test, future in zip(chunks_test, futures_all):
        result.append(darts.metrics.accuracy(test, future))
    avg = np.average(result)
    print(
        f"The average accuracy is {avg} for {get_print_string(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation, tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size)}")
    return result, avg

def reset_experiment(intermediate, exp):
    intermediate[exp] = get_intermediate(exp)


def segment_data_bundle(bundle, fc_cfg, tp_cfg, worker_bundle):
    # value_columns = [var for var in series_scaled.components.to_list() if var.startswith(SENSOR_PREFIX) or var == DIFFERENCE_TS]
    # series_scaled_segmented = segment_by_control_mode(series_scaled, MODE, value_columns, st_input_size)
    exp = fc_cfg.exp
    input_size = tp_cfg.input_size
    value_columns = [var for var in bundle.val_scaled.components.to_list() if
                     var.startswith(fc_cfg.sensor_prefix) or var == DIFFERENCE_TS]
    # TODO update for multiple control mode columns
    val_scaled_segmented = segment_by_control_mode(bundle.val_scaled, get_mode_name(exp, 0), value_columns, input_size)
    value_columns = [var for var in bundle.train_scaled.components.to_list() if
                     var.startswith(fc_cfg.sensor_prefix) or var == DIFFERENCE_TS]
    train_scaled_segmented = segment_by_control_mode(bundle.train_scaled, get_mode_name(exp, 0), value_columns, input_size)
    past_covariates = {mode: [ts[DIFFERENCE_TS] for ts in item] for mode, item in
                       train_scaled_segmented.items()}
    val_past_covariates = {mode: [ts[DIFFERENCE_TS] for ts in item] for mode, item in
                           val_scaled_segmented.items()}
    train_scaled_segmented = {
        mode: [ts[[s for s in bundle.train_scaled.components.to_list() if s.startswith(fc_cfg.sensor_prefix)]] for ts in item]
        for mode, item in train_scaled_segmented.items()}
    return train_scaled_segmented, val_scaled_segmented, past_covariates, val_past_covariates



def add_persistence_ts(ts: TimeSeries, start_value=None):
    """
    Takes a binary TimeSeries and returns a TimeSeries
    where each value is the number of steps since the last mode switch.
    """
    # 1. Get values as a numpy array
    vals = ts.values().flatten()

    # 2. Identify where switches occur (0->1 or 1->0)
    # We use a shift-and-compare logic
    switches = np.concatenate(([True], vals[1:] != vals[:-1]))

    # 3. Create the ramp (time in state)
    # cumsum() creates a group ID for each 'stretch' of the same mode
    group_ids = np.cumsum(switches)

    # Use a trick with cumcount in pandas for the fastest calculation
    # or a pure numpy approach:
    time_in_state = np.zeros(len(vals))
    count = 0
    for i in range(len(vals)):
        if switches[i]:
            count = 0
        else:
            count += 1
        time_in_state[i] = count

    # 4. Return as a Darts TimeSeries
    return TimeSeries.from_times_and_values(
        times=ts.time_index,
        values=time_in_state,
        columns=['time_in_state_feature']
    )


def add_dynamic_distance_multivariate(ts: TimeSeries,
                                      sensor_col: str,
                                      mode_col: str,
                                      threshold_low: float,
                                      threshold_high: float):
    """
    Calculates distance to thresholds from a multivariate TimeSeries.
    """
    # Access specific components
    s_vals = ts[sensor_col].values().flatten()
    m_vals = ts[mode_col].values().flatten()

    # Vectorized logic:
    # If mode is 0 (Low), we care about how close we are to threshold_high.
    # If mode is 1 (High), we care about how close we are to threshold_low.
    active_thresholds = np.where(m_vals == 0, threshold_high, threshold_low)

    # We remove np.abs() here to allow the Reconciler to see 'crossings'
    # (Negative values mean we've passed the threshold but log hasn't flipped)
    dist = s_vals - active_thresholds

    # Correct the sign so 'approaching' is always decreasing toward 0
    # For Low Mode (0), sensor is below threshold_high -> (s_vals - threshold_high) is negative.
    # We multiply by -1 if mode is 0 to make it a 'positive distance to trigger'.
    dist = np.where(
        m_vals == 0,

        s_vals - threshold_low,  # Mode 0
        threshold_high - s_vals  # Mode 1
    )

    return TimeSeries.from_times_and_values(
        times=ts.time_index,
        values=dist,
        columns=['dist_to_active_threshold']
    )


def get_prediction_uncertainty(prediction):
    prediction_slice = prediction.to_dataframe()
    spread = prediction_slice['sensor_temperature_q0.950'] - prediction_slice['sensor_temperature_q0.050']
    point = prediction_slice['sensor_temperature_q0.500']
    return prediction['sensor_temperature_q0.500'].with_values(spread), prediction[
        'sensor_temperature_q0.500'].with_values(point)


def shift_series(series, fill_value):
    component = series.components[0]
    shifted = np.concatenate([series[component].values()[1:], np.array([[fill_value]])])
    return series.with_values(shifted)

def truncate_to_prediction(series_list, input_size, train_spread=None):
    if train_spread is None:
        return [series[input_size:] for series in series_list]
    else:
        return [series[input_size:].drop_after(train_spread.end_time(), keep_point=True) for series in series_list]

def get_gridsearch_file_string(exp, fold, model_type, prediction_horizon, prediction_stride, input_size):
    return f"{PATH_PREFIX}/tmp/{exp}_{fold}_{model_type}_{prediction_horizon}_{prediction_stride}_{input_size}_gridsearch_upper.json"

def get_log_file_string(exp, fold, model_type, prediction_horizon, prediction_stride, input_size, multi_mode):
    return f"{PATH_PREFIX}/tmp/{exp}_{fold}_{model_type}_{prediction_horizon}_{prediction_stride}_{input_size}_{multi_mode}.out"

def get_futures_string(exp, fold, st_lt, model_type, prediction_horizon, prediction_stride, input_size, multi_mode, val=True):
    return f"{PATH_PREFIX}/data/futures/{exp}_{fold}_{st_lt}_{model_type}_{prediction_horizon}_{prediction_stride}_{input_size}_{multi_mode}{'' if not val else '_val'}.pkl"

def store_futures(futures, exp, fold, st_lt, model_type, prediction_horizon, prediction_stride, input_size, multi_mode, val):
    futures_path = get_futures_string(exp, fold, st_lt, model_type, prediction_horizon, prediction_stride, input_size, multi_mode, val)
    os.makedirs(os.path.dirname(futures_path), exist_ok=True)
    with open(futures_path, "wb") as f:
        pickle.dump(futures, f)

def get_futures( exp, fold, st_lt, model_type, prediction_horizon, prediction_stride, input_size, multi_mode, val):
    with open(get_futures_string(exp, fold, st_lt, model_type, prediction_horizon, prediction_stride, input_size, multi_mode), "rb") as f:
        futures = pickle.load(f)
    return futures


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_model_file_string(lower, upper, exp, fold, model_type, prediction_horizon, prediction_stride, input_size, multi):
    return f"{PATH_PREFIX}/models_tmp/LightGBM_gridsearch_training_l{lower}_u{upper}_{exp}_{fold}_{model_type}_{prediction_horizon}_{prediction_stride}_{input_size}_{multi}.pkl"



def build_ts_sim_from_covariates(bundle, last_control, lower, upper, fc_cfg, scale_min, scale_max, mode_component):
    all_past_covariates_sim = sim_past_covariates(
        concatenate([bundle.train_scaled, bundle.val_scaled, bundle.test_scaled]),
        lower, upper, scale_min, scale_max, fc_cfg, pad=True, last_control=last_control
    )
    df = pd.DataFrame(
        data=all_past_covariates_sim,
        columns=mode_component,
        index=pd.date_range(
            start=bundle.train_scaled.start_time(),
            periods=len(all_past_covariates_sim),
            freq=SAMPLING_RATE
        )
    )
    return TimeSeries.from_dataframe(df, value_cols=mode_component)


def encode_cfg(cfg):
    return base64.b64encode(cfg.to_json().encode('utf-8')).decode('utf-8')


def stack_covariates_for_multi(all_past_covariates_sim: TimeSeries, bundle, mode_component,
                               past_covariates_sim: TimeSeries, val_past_covariates_sim: TimeSeries) -> tuple[
    TimeSeries, TimeSeries, TimeSeries]:
    past_covariates = bundle.train_scaled[mode_component]
    val_past_covariates = bundle.val_scaled[mode_component]
    all_past_covariates = concatenate(
        [past_covariates, val_past_covariates, bundle.test_scaled[mode_component]])
    all_past_covariates_sim = all_past_covariates_sim.stack(all_past_covariates)
    past_covariates_sim = past_covariates_sim.stack(past_covariates)
    val_past_covariates_sim = val_past_covariates_sim.stack(val_past_covariates)
    return all_past_covariates_sim, past_covariates_sim, val_past_covariates_sim

def get_serial_t(model_type):
    return "pkl"



def select_best_model_by_rmse(all_intermediate, tp_cfg, fc_cfg, allow_simulation=True):
    exp = fc_cfg.exp
    model_val = fc_cfg.model_type.value
    component = tp_cfg.component.value
    st_lt = Horizon.ST.value
    hor = str(tp_cfg.prediction_horizon)
    size = str(tp_cfg.input_size)
    stride = str(tp_cfg.prediction_stride)
    rmse_key = Scores.RMSE.value

    if tp_cfg.prediction_stride == RUN_PREDICTION_STRIDES[exp][1]:
        stride = str(RUN_PREDICTION_STRIDES[exp][0])

    results = []

    # Check Segmentation OFF
    try:
        off_res = \
        all_intermediate[exp][model_val][Segmentation.OFF.value][component][st_lt][hor][size][stride]['scores'][
            rmse_key]
        if off_res is not None:
            results.append({'seg': Segmentation.OFF, 'mode': None, 'avg_rmse': np.average(off_res)})
        else:
            results.append({'seg': Segmentation.OFF, 'mode': None, 'avg_rmse': 1.0})
    except (KeyError, TypeError):
        pass

    # Check Segmentation ON (modes '0.0' and '1.0')
    if allow_simulation:
        for mode in ['0.0', '1.0']:
            try:
                on_res = \
                all_intermediate[exp][model_val][Segmentation.ON.value][component][mode][hor][size][stride]['scores'][
                    rmse_key]
                if on_res is not None:
                    results.append({'seg': Segmentation.ON, 'mode': mode, 'avg_rmse': np.average(on_res)})
            except (KeyError, TypeError):
                continue

    if not results:
        print("No results found in all_intermediate for the given configuration.")
        return None

    # Select the one with the minimum average RMSE
    best = min(results, key=lambda x: x['avg_rmse'])

    print(
        f"Best model for {exp}: Segmentation {best['seg'].value}, Mode {best['mode']} with Avg RMSE: {best['avg_rmse']:.4f}")
    return best, results


def get_scenario_exp(exp) -> Any:
    return exp + "_scenario"

def get_superprocess_exp(exp: str) -> str:
    return get_scenario_exp(exp) if RUN_SCENARIO else exp

def get_hdt_eval_path(exp, model_type_str: str | Any, file, allow_simulation: bool = False):
    exp_for_superprocess = get_superprocess_exp(exp)
    effective_model_type_str = model_type_str + "_sim" if allow_simulation else model_type_str
    url_file = os.path.join(
        get_hdt_data_path(exp_for_superprocess),
        "superprocess_log",
        effective_model_type_str,
        file
    )
    return url_file


def get_hdt_eval_file(exp, model_type, file, allow_simulation: bool = False) -> str:
    model_type_str = model_type.value if hasattr(model_type, "value") else str(model_type)
    index_file = get_hdt_eval_path(exp, model_type_str, file, allow_simulation=allow_simulation)
    return index_file


def construct_worker_bundle(bundle, chunks: list[Any], mode_component) -> WorkerBundle:
    past_covariates = bundle.train_scaled[mode_component]
    val_past_covariates = bundle.val_scaled[mode_component]
    all_past_covariates = concatenate([past_covariates, val_past_covariates, bundle.test_scaled[mode_component]])
    w_bundle = WorkerBundle(chunks, past_covariates, val_past_covariates, all_past_covariates)
    return w_bundle


def construct_sensor_chunks(bundle, fc_cfg, tp_cfg) -> list[Any]:
    chunk_duration = pd.Timedelta(tp_cfg.prediction_horizon * SAMPLING_FACTOR, "s")
    prediction_stride_duration = pd.Timedelta(tp_cfg.prediction_stride * SAMPLING_FACTOR, "s")
    test_scaled_sensors = get_sensors_only(bundle.test_scaled, fc_cfg)
    chunks = chunk_series(test_scaled_sensors, chunk_duration, prediction_stride_duration)
    return chunks


def load_or_train(best_lower: float, best_upper: float, bundle, component: Literal[Components.ALL], exp, fc_cfg,
                  input_size, model_type, multi_mode: str | Literal['1.0'], parameters,
                  segmentation: Literal[Segmentation.ON], st_lt: Literal[Horizon.ST], tp_cfg: TrainPredictConfig,
                  w_bundle: WorkerBundle) -> tuple[str, ForecastingModel | TorchForecastingModel]:
    file_name = get_model_file_string(best_lower, best_upper, exp, tp_cfg.fold, model_type,
                                      tp_cfg.prediction_horizon,
                                      tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)

    file_name_int = get_model_file_string(int(best_lower), int(best_upper), exp, tp_cfg.fold, model_type,
                                          tp_cfg.prediction_horizon,
                                          tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)
    model_file = get_simple_model_path(exp, model_type, st_lt, component, segmentation,
                                       tp_cfg.prediction_horizon,
                                       input_size, get_serial_t(fc_cfg.model_type), mode=multi_mode)
    constructor = get_constructor(model_type)
    load_kwargs = {"weights_only": False} if model_type in [Models.LSTM, Models.DLinear,
                                                            Models.TCN] and torch.cuda.is_available() else {}
    # load_kwargs = {}
    if os.path.exists(file_name):

        model = constructor.load(file_name, **load_kwargs)
    elif os.path.exists(model_file):
        model = constructor.load(model_file, **load_kwargs)
    elif os.path.exists(file_name_int):
        model = constructor.load(file_name_int, **load_kwargs)
    else:
        model = constructor(**parameters)
        model.fit(series=get_sensors_only(bundle.train_scaled, fc_cfg),
                  val_series=get_sensors_only(bundle.val_scaled, fc_cfg),
                  past_covariates=w_bundle.past_covariates,
                  val_past_covariates=w_bundle.val_past_covariates)
        model.save(file_name)
        convert_and_store_onnx_model(exp, model, fc_cfg.model_type, st_lt, component, segmentation,
                                     tp_cfg.prediction_horizon, input_size, multi_mode)
    return model, model_file




def replace_last_values(
    ts_old: TimeSeries,
    ts_new: TimeSeries,
    expected_n
) -> TimeSeries:
    # Basic sanity check
    if len(ts_new) > len(ts_old):
        raise ValueError("ts new is longer than ts old; cannot replace last part.")

    v = ts_old.values(copy=True)
    v2 = ts_new.values(copy=True)

    if v.shape[1:] != v2.shape[1:]:
        raise ValueError(
            f"Value shapes (excluding time) do not match: {v.shape[1:]} vs {v2.shape[1:]}"
        )

    #n = len(ts_new)
    n = len(ts_old) - expected_n
    v[-n:, :] = v2[:n]

    return ts_old.with_values(v)

