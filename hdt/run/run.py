import gc
import os.path
from itertools import product
from typing import List

from darts.models import DLinearModel, LightGBMModel
from joblib import Parallel, delayed
import math
import subprocess

from hdt.util import *
from hdt.cleaned_cpee_approach import predict_simulative_control
from hdt.preprocessing.extract_timeseries import DIFFERENCE_TS
from hdt.preprocessing.preprocess import preprocess_exp, load_or_prepare_scenario_bundle
from hdt.parameters import COARSE_BOUNDARIES, Components, Horizon, Segmentation
from hdt.run.configuration import SeriesBundle, WorkerBundle, TrainPredictConfig, EvalConfig, ForecastConfig


runtime_pattern = r"Runtime:\s*(?P<runtime>\d+(?::\d+){2}(?:\.\d+)?)"
pattern = (
    r"Best upper lower\s*\n"
    r"Lower bound:\s*(?P<lower>[\d.]+)\s*\n"
    r"Upper bound:\s*(?P<upper>[\d.]+)\s*\n"
    r"Validation RMSE:\s*(?P<rmse>[\d.]+)\s*\n"
)

def configured_train_predict_evaluate(configuration):
    def train_predict_evaluate(exp,
                               st_lt,
                               all_intermediate,
                               fold):
        if GRIDSEARCH_PARAM in configuration:
            gs = True
            # base_parameters = configuration[GRIDSEARCH_PARAM]
        else:
            gs = False
        forecast_config = ForecastConfig(exp, configuration[TYPE], st_lt, SENSOR_PREFIX, MODE_PREFIX, DIFFERENCE_TS,
                                         get_all_mode_names(exp))

        bundle_file = get_bundles_path(exp, fold)
        if not os.path.exists(bundle_file):
            series_scaled, train_scaled, val_scaled, test_scaled, scale_min, scale_max = preprocess_exp(exp, fold)
            data_bundle = SeriesBundle(train_scaled, val_scaled, test_scaled)
            with open(bundle_file, "wb") as f:
                pickle.dump((data_bundle, scale_min, scale_max), f)
        else:
            with open(bundle_file, "rb") as f:
                data_bundle, scale_min, scale_max = pickle.load(f)
                data_bundle = SeriesBundle(
                    data_bundle.train_scaled.astype(np.float32),
                    data_bundle.val_scaled.astype(np.float32),
                    data_bundle.test_scaled.astype(np.float32)
                )
        print(
            f"Sanity checking run: {forecast_config} {data_bundle.train_scaled.start_time()} {data_bundle.train_scaled.end_time()} {data_bundle.train_scaled.freq}")
        cpee_predictive_simulation = {}

        eval_cfg = EvalConfig(scale_min, scale_max)
        ### 2: Components
        for component in RUN_COMPONENTS:
            ### 3: Prediction Horizon
            for prediction_horizon in RUN_PREDICTION_HORIZONS[exp]:
                ### 4: Prediction Stride
                for prediction_stride in RUN_PREDICTION_STRIDES[exp] if not SKIP_LONG_STRIDES else [RUN_PREDICTION_STRIDES[exp][0]]:
                    ### 5: Input size
                    for input_size in RUN_INPUT_SIZES[exp][prediction_horizon]:
                        exp_boundaries = COARSE_BOUNDARIES.get(exp)
                        if st_lt is Horizon.ST:
                            parameters = configuration[P_GET](input_size, 1, component is Components.ALL)
                            if component is Components.ALL:
                                tp_cfg = TrainPredictConfig(component,
                                                            fold,
                                                            bundle_file,
                                                            prediction_horizon,
                                                            prediction_stride,
                                                            input_size,
                                                            gs=gs,
                                                            sim=True,
                                                            train_boundaries=True,
                                                            lower_vals=exp_boundaries["min_vals"],
                                                            upper_vals=exp_boundaries["max_vals"],
                                                            actual_boundaries=ACTUAL_BOUNDARIES[exp]
                                                            )
                            else:
                                tp_cfg = TrainPredictConfig(component,
                                                            fold,
                                                            bundle_file,
                                                            prediction_horizon,
                                                            prediction_stride,
                                                            input_size,
                                                            gs,
                                                            sim=False,
                                                            lower_vals=exp_boundaries["min_vals"],
                                                            upper_vals=exp_boundaries["max_vals"],
                                                            train_boundaries=False
                                                            )
                        else:
                            parameters = configuration[P_GET](input_size, prediction_horizon,
                                                              component is Components.ALL)
                            tp_cfg = TrainPredictConfig(component,
                                                        fold,
                                                        bundle_file,
                                                        prediction_horizon,
                                                        prediction_stride,
                                                        input_size,
                                                        gs,
                                                        sim=False,
                                                        lower_vals=exp_boundaries["min_vals"],
                                                        upper_vals=exp_boundaries["max_vals"],
                                                        train_boundaries=False
                                                        )
                        try:
                            # LT TCN models cannot handle equal input size and prediction horizon by construction
                            if not (
                                        (forecast_config.model_type is Models.TCN and st_lt is Horizon.LT and prediction_horizon == input_size) or
                            # ST TCN models cannot handle the irrigation 24 input size case, because the kernel size is 25 and must be strictly smaller than the input size
                                        (forecast_config.model_type is Models.TCN and st_lt is Horizon.ST and exp == IRRIGATION and input_size == 24)
                                ):
                                tp_cfg, w_bundle = run_train_predict_evaluate(all_intermediate, data_bundle,
                                                                              forecast_config, eval_cfg, tp_cfg,
                                                                              parameters)
                                if component is Components.ALL and st_lt is Horizon.ST:
                                    if tp_cfg.lower is None and tp_cfg.upper is None:
                                        # Find the corresponding key with the same prediction_horizon and input_size
                                        for key in cpee_predictive_simulation:
                                            if key[1] == tp_cfg.prediction_horizon and key[2] == tp_cfg.input_size:
                                                tp_cfg = replace(tp_cfg, lower=key[3], upper=key[4])
                                                w_bundle = cpee_predictive_simulation[key][1]
                                                break

                                    cpee_predictive_simulation[
                                        (tp_cfg.prediction_stride, tp_cfg.prediction_horizon, tp_cfg.input_size,
                                         tp_cfg.lower, tp_cfg.upper)] = (tp_cfg,
                                                                         w_bundle) if st_lt is Horizon.ST and component is Components.ALL else None
                        except Exception as e:
                            print(
                                f"Error {e} occured for {forecast_config} {tp_cfg} {eval_cfg}. Continuing...")
                            traceback.print_exc()
        def _ensure_best_models_onnx_serialized():
            # Only relevant for the CPEE path (ST + ALL), because that's what produces cpee_predictive_simulation.
            if not (st_lt is Horizon.ST and len(cpee_predictive_simulation) > 0):
                return

            for _comb, (tp_cfg, _w_bundles) in cpee_predictive_simulation.items():
                res = select_best_model_by_rmse(all_intermediate, tp_cfg, forecast_config)
                if res is None:
                    # No "best model" recorded; nothing to serialize.
                    continue

                best, _results = res
                if best is None:
                    continue


                seg = best["seg"]
                mode = best["mode"]

                onnx_path = get_simple_model_path(
                    exp=forecast_config.exp,
                    mt=forecast_config.model_type,
                    st_lt=forecast_config.st_lt,
                    component=tp_cfg.component,
                    segmentation=seg,
                    prediction_horizon=tp_cfg.prediction_horizon,
                    input_size=tp_cfg.input_size,
                    serialization_t="onnx",
                    mode=mode,
                )
                if os.path.exists(onnx_path):
                    continue

                pkl_path = get_simple_model_path(
                    exp=forecast_config.exp,
                    mt=forecast_config.model_type,
                    st_lt=forecast_config.st_lt,
                    component=tp_cfg.component,
                    segmentation=seg,
                    prediction_horizon=tp_cfg.prediction_horizon,
                    input_size=tp_cfg.input_size,
                    serialization_t=get_serial_t(forecast_config.model_type),
                    mode=mode,
                )

                if not os.path.exists(pkl_path):
                    print(
                        f"Warning: Cannot create ONNX model because source model file is missing: {pkl_path}"
                    )
                    continue

                try:
                    constructor = get_constructor(forecast_config.model_type)
                    load_kwargs = {"weights_only": False} if forecast_config.model_type in [
                        Models.LSTM, Models.DLinear, Models.TCN
                    ] and torch.cuda.is_available() else {}
                    model = constructor.load(pkl_path, **load_kwargs)
                    print("Converting to ONNX...")
                    success = convert_and_store_onnx_model(
                            exp=forecast_config.exp,
                            model=model,
                            model_type=forecast_config.model_type,
                            st_lt=forecast_config.st_lt,
                            component=tp_cfg.component,
                            segmentation=seg,
                            prediction_horizon=tp_cfg.prediction_horizon,
                            input_size=tp_cfg.input_size,
                            mode=mode,
                        )
                    if not success:
                        # Load chkpt + pkl files from models_tmp
                        multi_mode = "0.0"
                        segmentation = Segmentation.ON
                        model_type = forecast_config.model_type
                        file_name = get_model_file_string(tp_cfg.lower, tp_cfg.upper, exp, tp_cfg.fold, model_type,
                                                          tp_cfg.prediction_horizon,
                                                          tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)

                        file_name_int = get_model_file_string(int(tp_cfg.lower), int(tp_cfg.upper), exp, tp_cfg.fold,
                                                              model_type,
                                                              tp_cfg.prediction_horizon,
                                                              tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)
                        load_kwargs = {"weights_only": False} if model_type in [Models.LSTM, Models.DLinear,
                                                                                Models.TCN] and torch.cuda.is_available() else {}
                        # load_kwargs = {}
                        if os.path.exists(file_name):
                            model = constructor.load(file_name, **load_kwargs)
                        elif os.path.exists(file_name_int):
                            model = constructor.load(file_name_int, **load_kwargs)
                        else:
                            model = constructor(**parameters)
                            model.fit(series=get_sensors_only(data_bundle.train_scaled, forecast_config),
                                      val_series=get_sensors_only(data_bundle.val_scaled, forecast_config),
                                      past_covariates=w_bundle.past_covariates,
                                      val_past_covariates=w_bundle.val_past_covariates)
                            model.save(file_name)
                            convert_and_store_onnx_model(exp, model, forecast_config.model_type, st_lt, component, segmentation,
                                                         tp_cfg.prediction_horizon, input_size, multi_mode)
                        model.save(get_simple_model_path(exp, model_type, st_lt, component, segmentation, prediction_horizon, input_size, mode=mode), clean=True)
                        success = convert_and_store_onnx_model(
                            exp=forecast_config.exp,
                            model=model,
                            model_type=forecast_config.model_type,
                            st_lt=forecast_config.st_lt,
                            component=tp_cfg.component,
                            segmentation=seg,
                            prediction_horizon=tp_cfg.prediction_horizon,
                            input_size=tp_cfg.input_size,
                            mode=mode,
                        )
                        if not success:
                            print(f"Warning: ONNX conversion did not create expected file: {onnx_path}")
                    #if not os.path.exists(onnx_path):
                    #    print(f"Warning: ONNX conversion did not create expected file: {onnx_path}")
                except Exception as e:
                    print(f"Warning: Failed to ensure ONNX serialization for {forecast_config} {tp_cfg}: {e}")
                    traceback.print_exc()

        _ensure_best_models_onnx_serialized()
        print(f"Finished {forecast_config}: produced {len(cpee_predictive_simulation)} CPEE configs")
        return cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg
    return train_predict_evaluate




def run_train_predict_evaluate(all_intermediate, bundle, fc_cfg, eval_cfg, tp_cfg, parameters):  # the extraction of the predicted series (chunked per stride) currently still needs the test with control modes (for preparing the forecast input)

    # Chunking must be done by the individual workers
    # Similarly the past_covariates
    chunks = construct_sensor_chunks(bundle, fc_cfg, tp_cfg)
    #chunks_no_stride = chunk_series(test_scaled_sensors, chunk_duration)
    # actual_test_scaled = concatenate(chunks)
    st_lt = fc_cfg.st_lt
    component = tp_cfg.component
    mode_component = fc_cfg.mode_component
    input_size = tp_cfg.input_size
    segmentation = tp_cfg.segmentation
    model_type = fc_cfg.model_type
    if tp_cfg.gs:
        parameters["random_state"] = [random_state]
    else:
        #if model_type in [Models.LightGBM, Models.DLinear, Models.LSTM]:
        parameters["random_state"] = random_state
        #else:
        #    parameters["random_seed"] = random_state

    # Prepare worker bundle
    if component is Components.SENSORS_ONLY:
        past_covariates = None
        val_past_covariates = None
        all_past_covariates = None
        w_bundle = WorkerBundle(chunks, past_covariates, val_past_covariates, all_past_covariates)
        w_bundles = [w_bundle]
        model = start_training(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode=None)
        if not check_existing_model(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation,
                                    tp_cfg.prediction_horizon,
                                    input_size=tp_cfg.input_size, serialization_t="onnx",
                                    mode=None):
            convert_and_store_onnx_model(fc_cfg.exp, model, fc_cfg.model_type, st_lt, component, segmentation,
                                     tp_cfg.prediction_horizon, input_size, None)
    elif st_lt is Horizon.LT or component is Components.ALL: # Train also the real control modes continued with CPEE here
        w_bundle = construct_worker_bundle(bundle, chunks, mode_component)
        bundle = SeriesBundle(
            get_sensors_only(bundle.train_scaled, fc_cfg).astype(np.float32),
            get_sensors_only(bundle.val_scaled, fc_cfg).astype(np.float32),
            get_sensors_only(bundle.test_scaled, fc_cfg).astype(np.float32)
        )
        w_bundles = [w_bundle]
        model = start_training(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode=None)
        if st_lt is Horizon.ST:
            if not check_existing_model(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component,
                                        tp_cfg.segmentation,
                                        tp_cfg.prediction_horizon,
                                        input_size=tp_cfg.input_size, serialization_t="onnx",
                                        mode=None):
                convert_and_store_onnx_model(fc_cfg.exp, model, fc_cfg.model_type, st_lt, component, segmentation,
                                             tp_cfg.prediction_horizon, input_size, None)
    if st_lt is Horizon.ST and component is Components.ALL and fc_cfg.exp != SWAT and RUN_PREDICTION_STRIDES[fc_cfg.exp][1] != tp_cfg.prediction_stride:
        # SWAT dataset does not have gridsearch on data condition boundaries
        # Additionally, training with both real control modes as covariates continued with the actual boundaries
        # plus the covariates chosen from the max and min val boundaries of the tp_cfg ranges
        segmentation = Segmentation.ON
        multi_mode = '0.0'
        model, tp_cfg, w_bundles, best_rmse = run_parallelized_train(all_intermediate, bundle, chunks, component,
                                                         eval_cfg, fc_cfg, input_size, mode_component,
                                                         model_type, multi_mode, parameters, segmentation, st_lt,
                                                         tp_cfg, w_bundles)
        #tp_cfg = replace(tp_cfg, multi=True)
        #multi_mode = '1.0'
        #model, tp_cfg, w_bundles = run_parallelized_train(all_intermediate, bundle, chunks, chunks_no_stride, component,
        #                                                 eval_cfg, fc_cfg, input_size, mode_component,
        #                                                 model_type, multi_mode, parameters, segmentation, st_lt,
        #                                                 tp_cfg, w_bundles)

    if ALWAYS_EVAL:
        if fc_cfg.st_lt is Horizon.ST and component is Components.ALL and fc_cfg.exp != SWAT and RUN_PREDICTION_STRIDES[fc_cfg.exp][1] != tp_cfg.prediction_stride:

                for stride in RUN_PREDICTION_STRIDES[fc_cfg.exp]:
                    # First
                    try:
                        # First the real control modes
                        if RUN_REAL_SIMULATED[0]:
                            futures_file = get_futures_string(fc_cfg.exp, tp_cfg.fold, fc_cfg.st_lt, fc_cfg.model_type,
                                          tp_cfg.prediction_horizon, stride, tp_cfg.input_size, None, val=EVAL_ON_VAL)
                            if not os.path.exists(futures_file):
                                if stride == RUN_PREDICTION_STRIDES[fc_cfg.exp][1]:
                                    # for the larger stride, the chunks must be properly computed
                                    chunk_duration = pd.Timedelta(tp_cfg.prediction_horizon * SAMPLING_FACTOR, "s")
                                    prediction_stride_duration = pd.Timedelta(stride * SAMPLING_FACTOR,
                                                                              "s")
                                    chunks = chunk_series(bundle.test_scaled, chunk_duration, prediction_stride_duration)
                                    w_bundle = replace(w_bundles[0], chunks=chunks)
                                    w_bundles[0] = w_bundle
                                else:
                                    w_bundle = w_bundles[0]
                                simulated_covariates = (w_bundle.all_past_covariates, None)
                                tp_cfg = replace(tp_cfg, sim=False)
                                load_kwargs = {"weights_only": False} if fc_cfg.model_type in [Models.LSTM, Models.DLinear, Models.TCN] and torch.cuda.is_available() else {}
                                run_prediction(all_intermediate, bundle, eval_cfg, fc_cfg, load_kwargs, None,
                                               segmentation.OFF, simulated_covariates, stride, tp_cfg, w_bundle, tp_cfg.actual_boundaries[0], tp_cfg.actual_boundaries[1], val=EVAL_ON_VAL)
                        elif RUN_REAL_SIMULATED[1]:
                            multi_mode = "0.0"
                            futures_file = get_futures_string(fc_cfg.exp, tp_cfg.fold, fc_cfg.st_lt, fc_cfg.model_type,
                                                              tp_cfg.prediction_horizon, stride, tp_cfg.input_size, multi_mode, val=EVAL_ON_VAL)
                            if EVAL_ON_VAL:
                                if not os.path.exists(futures_file) and best_rmse is None:
                                    w_bundle = w_bundles[1]
                                    simulated_covariates = (w_bundle.all_past_covariates, None)
                                    tp_cfg = replace(tp_cfg, sim=True)
                                    load_kwargs = {"weights_only": False} if fc_cfg.model_type in [Models.LSTM, Models.DLinear,
                                                                                                   Models.TCN] and torch.cuda.is_available() else {}
                                    segmentation = Segmentation.ON
                                    run_prediction(all_intermediate, bundle, eval_cfg, fc_cfg, load_kwargs, multi_mode,
                                                   segmentation, simulated_covariates, stride, tp_cfg, w_bundle, tp_cfg.lower, tp_cfg.upper, val=EVAL_ON_VAL)
                                elif best_rmse is not None:
                                    assign_intermediate(all_intermediate, fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt,
                                                        segmentation, tp_cfg.component,
                                                        [best_rmse], tp_cfg.prediction_horizon, tp_cfg.prediction_stride,
                                                        tp_cfg.input_size,
                                                        score_key=Scores.RMSE.value, mode=multi_mode)
                                store_intermediate(all_intermediate)
                            else:
                                # Predict on test
                                if not os.path.exists(futures_file):
                                    if stride == RUN_PREDICTION_STRIDES[fc_cfg.exp][1]:
                                        # for the larger stride, the chunks must be properly computed
                                        chunk_duration = pd.Timedelta(tp_cfg.prediction_horizon * SAMPLING_FACTOR, "s")
                                        prediction_stride_duration = pd.Timedelta(stride * SAMPLING_FACTOR,
                                                                                  "s")
                                        chunks = chunk_series(bundle.test_scaled, chunk_duration,
                                                              prediction_stride_duration)
                                        w_bundle = replace(w_bundles[1], chunks=chunks)
                                        w_bundles[1] = w_bundle
                                    else:
                                        w_bundle = w_bundles[1]
                                    simulated_covariates = (w_bundle.all_past_covariates, None)
                                    tp_cfg = replace(tp_cfg, sim=True)
                                    load_kwargs = {"weights_only": False} if fc_cfg.model_type in [Models.LSTM,
                                                                                                   Models.DLinear,
                                                                                                   Models.TCN] and torch.cuda.is_available() else {}
                                    run_prediction(all_intermediate, bundle, eval_cfg, fc_cfg, load_kwargs, multi_mode,
                                                   segmentation.ON, simulated_covariates, stride, tp_cfg, w_bundle,
                                                   tp_cfg.lower, tp_cfg.upper,
                                                   val=EVAL_ON_VAL)
                    except Exception as e:
                        print(
                            f"Error {e} in predicting simulated futures for {fc_cfg} {tp_cfg} {eval_cfg} {stride}. Continuing...")
                        traceback.print_exc()
        elif fc_cfg.st_lt is Horizon.LT and component is Components.ALL:
            run_evaluation(all_intermediate, fc_cfg, tp_cfg, bundle, w_bundle, model, eval_cfg)
            if RUN_SCENARIO:
                bundle_scenario = load_or_prepare_scenario_bundle(fc_cfg, eval_cfg, bundle, tp_cfg.fold)
                chunks_scenario = construct_sensor_chunks(bundle, fc_cfg, tp_cfg)
                w_bundle_scenario = construct_worker_bundle(bundle_scenario, chunks_scenario, mode_component)
                run_evaluation(all_intermediate, fc_cfg, tp_cfg, bundle_scenario, w_bundle_scenario, model, eval_cfg, "scenario")
    return tp_cfg, w_bundles


def run_evaluation(all_intermediate, fc_cfg, tp_cfg, bundle, w_bundle, model, eval_cfg, multi_mode=None):
    futures_file = get_futures_string(fc_cfg.exp, tp_cfg.fold, fc_cfg.st_lt, fc_cfg.model_type,
                                      tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size, multi_mode,
                                      val=False)
    if not os.path.exists(futures_file):
        pred_future = predict(all_intermediate, bundle, fc_cfg, tp_cfg, w_bundle, model)
        store_futures(pred_future, fc_cfg.exp, tp_cfg.fold, fc_cfg.st_lt, fc_cfg.model_type,
                      tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size, multi_mode, val=False)
        if multi_mode is not None:
            tp_cfg = replace(tp_cfg, component=Components.SENSORS_ONLY) ## Encode the scenario evaluation results as Component.SENSORS_ONLY to avoid recomputing the whole all_intermediate json
        evaluate(all_intermediate, pred_future, fc_cfg, eval_cfg, tp_cfg, w_bundle)

def run_prediction(all_intermediate, bundle, eval_cfg, fc_cfg,
                   load_kwargs: dict[str, bool] | dict[Any, Any], multi_mode,
                   segmentation, simulated_covariates: tuple[Any, None], stride: int,
                   tp_cfg, w_bundle: WorkerBundle, lower, upper, val=False):
    model = get_constructor(fc_cfg.model_type).load(
        get_simple_model_path(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component,
                              segmentation, tp_cfg.prediction_horizon, tp_cfg.input_size,
                              get_serial_t(fc_cfg.model_type), mode=multi_mode), **load_kwargs)
    if not val:
        chunks_test = w_bundle.chunks
    else:
        chunk_duration = pd.Timedelta(tp_cfg.prediction_horizon * SAMPLING_FACTOR, "s")
        stride_duration = pd.Timedelta(stride * SAMPLING_FACTOR, "s")
        chunks_test = chunk_series(bundle.val_scaled, chunk_duration, stride_duration)
    futures = predict_with_simulated_control(chunks_test, bundle, w_bundle, fc_cfg,
                                             tp_cfg, lower,
                                             upper, eval_cfg, model,
                                             simulated_covariates, val=val)
    store_futures(futures, fc_cfg.exp, tp_cfg.fold, fc_cfg.st_lt, fc_cfg.model_type,
                  tp_cfg.prediction_horizon, stride, tp_cfg.input_size, multi_mode, val=val)
    avgs = quick_evaluation(chunks_test, futures, fc_cfg, tp_cfg)
    assign_intermediate(all_intermediate, fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt,
                        segmentation, tp_cfg.component,
                        [avgs[1]], tp_cfg.prediction_horizon, tp_cfg.prediction_stride,
                        tp_cfg.input_size,
                        score_key=Scores.RMSE.value, mode=multi_mode)
    store_intermediate(all_intermediate)


def predict_with_simulated_control(chunks_test, bundle, w_bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model,
                                   simulated_covariates, val=False):

    if not val:
        base_time = bundle.val_scaled.end_time()
        past_series = concatenate([concatenate([bundle.val_scaled, bundle.test_scaled]), w_bundle.all_past_covariates.drop_before(bundle.train_scaled.end_time())], axis=1)
    else:
        base_time = bundle.train_scaled.end_time()
        past_series = concatenate([concatenate([bundle.train_scaled, bundle.val_scaled]),
                                   w_bundle.all_past_covariates.drop_after(bundle.val_scaled.end_time(), keep_point=True)], axis=1)
    return predict_simulative_control(
        chunks_test, bundle.train_scaled if val else bundle.val_scaled, fc_cfg, tp_cfg, eval_cfg, model,
        past_series, upper, lower, base_time,
        simulated_covariates=simulated_covariates
    )

def run_parallelized_train(all_intermediate, bundle, chunks: list[Any],
                           component: Literal[Components.ALL], eval_cfg, fc_cfg, input_size, mode_component, model_type, multi_mode: str, parameters,
                           segmentation: Literal[Segmentation.ON], st_lt: Literal[Horizon.ST], tp_cfg: TrainPredictConfig, w_bundles: List[WorkerBundle | Any]) -> \
tuple[Any | None, TrainPredictConfig, list[WorkerBundle | Any], float | None]:
    print(f"Consistency check. Sim: {tp_cfg.sim}, Train boundaries: {tp_cfg.train_boundaries}, mode {multi_mode}, and tp {tp_cfg}")
    fc_string = encode_cfg(fc_cfg)
    tp_string = encode_cfg(tp_cfg)
    eval_string = encode_cfg(eval_cfg)
    result_log = get_log_file_string(fc_cfg.exp, tp_cfg.fold, model_type, tp_cfg.prediction_horizon,
                                     tp_cfg.prediction_stride, input_size, multi_mode)
    print(result_log)
    if not os.path.exists(result_log):
        cmd = ["python", "-u", "-m", "hdt.run.parallelized_train", fc_string, tp_string, eval_string]

        #if True:
        captured_lines = []
        with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout to catch everything
                text=True,
                bufsize=1  # Line buffered
        ) as process:
            # Read from stdout in real-time
            for line in process.stdout:
                print(line, end="")  # Print to calling stdout
                captured_lines.append(line)  # Save for later use

           # Ensure the process finished correctly
            process.wait()


        out_log = "".join(captured_lines)
        with open(result_log, "w") as f:
            f.write(out_log)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        #else:
        #    result = subprocess.run(cmd, capture_output=True, check=True, text=True)
        #    out_log = result.stdout
    else:
        with open(result_log, "r") as f:
            out_log = f.read()

    match_run = re.search(runtime_pattern, out_log)

    if match_run:
        runtime_delta = pd.Timedelta(match_run.group("runtime"))
        print(f"Total seconds: {runtime_delta.total_seconds()}")
    else:
        runtime_delta = None
    match = re.search(pattern, out_log)
    if match:
        best_lower = float(match.group("lower"))
        best_upper = float(match.group("upper"))
        best_rmse = float(match.group("rmse"))
        print(f"Extracted from gridsearch: Lower={best_lower}, Upper={best_upper}, RMSE={best_rmse}")
        exp = fc_cfg.exp
        tp_cfg = replace(tp_cfg, lower=best_lower, upper=best_upper)
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component,
                            [best_rmse], tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size, score_key=Scores.RMSE.value, mode=multi_mode)
        sim_mode_component = get_all_sim_mode_components(exp)
        last_control = w_bundles[0].past_covariates[get_all_mode_names(fc_cfg.exp)].first_value()
        all_past_covariates_sim = build_ts_sim_from_covariates(bundle, last_control, best_lower, best_upper, fc_cfg,
                                                           eval_cfg.scale_min, eval_cfg.scale_max, sim_mode_component)
        past_covariates_sim = all_past_covariates_sim.drop_after(bundle.val_scaled.start_time())
        val_past_covariates_sim = all_past_covariates_sim.drop_after(bundle.val_scaled.end_time(),
                                                             keep_point=True).drop_before(
            bundle.train_scaled.end_time())
        if multi_mode == '1.0':
            print("Prepraring stacked covariates ... ")
            all_past_covariates_sim, past_covariates_sim, val_past_covariates_sim = stack_covariates_for_multi(
                all_past_covariates_sim, bundle, mode_component, past_covariates_sim, val_past_covariates_sim)
        w_bundle = WorkerBundle(chunks, past_covariates_sim, val_past_covariates_sim, all_past_covariates_sim)
        w_bundles.append(w_bundle)
        model, model_file = load_or_train(best_lower, best_upper, bundle, component, exp, fc_cfg, input_size,
                                          model_type, multi_mode, parameters, segmentation, st_lt, tp_cfg, w_bundle)
        tp_dict = tp_cfg.to_dict()
        lower_upper = {k: tp_dict[k] for k in ('lower', 'upper') if k in tp_dict}
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, lower_upper,
                            tp_cfg.prediction_horizon, tp_cfg.prediction_stride, input_size,
                            result_key=params_key, mode=multi_mode)
        print(
            f"Finished training for {exp} {model_type} {st_lt.value} {component.value} {multi_mode} training time {runtime_delta if runtime_delta is not None else ''}...\n")
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, runtime_delta,
                            tp_cfg.prediction_horizon, tp_cfg.prediction_stride,
                            input_size,
                            result_key=duration_t, mode=multi_mode)

        if not os.path.exists(model_file):
            store_model(exp, model, model_type, st_lt, component, segmentation, tp_cfg.prediction_horizon,
                        input_size,
                        mode=multi_mode)
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component,
                            model_file,
                            tp_cfg.prediction_horizon, tp_cfg.prediction_stride,
                            input_size,
                            result_key=actual_model, mode=multi_mode)
        if DEBUG_RUN:
            test_prediction(all_intermediate, all_past_covariates_sim, bundle, eval_cfg, fc_cfg, model, multi_mode,
                            tp_cfg, w_bundle)

    else:
        model = None
        w_bundles.append(None)
        best_rmse = None
        if DEBUG_RUN:
            best_lower = 81.2
            best_upper = 97.6
            file_name = get_model_file_string(best_lower, best_upper, fc_cfg.exp, tp_cfg.fold, model_type,
                                              tp_cfg.prediction_horizon,
                                              tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)
            constructor = get_constructor(model_type)
            if os.path.exists(file_name):
                model = constructor.load(file_name)
                base_time = bundle.val_scaled.end_time()
                past_series = concatenate([bundle.val_scaled, bundle.test_scaled])
                sim_mode_component = get_all_sim_mode_components(fc_cfg.exp)
                last_control = w_bundles[0].past_covariates[get_all_mode_names(fc_cfg.exp)].first_value()

                all_past_covariates_sim = build_ts_sim_from_covariates(bundle, last_control, best_lower, best_upper, fc_cfg,
                                                                       eval_cfg.scale_min, eval_cfg.scale_max,
                                                                       sim_mode_component)
                w_bundle = WorkerBundle(chunks, None, None,
                                        all_past_covariates_sim)
                test_prediction(all_intermediate, all_past_covariates_sim, bundle, eval_cfg, fc_cfg, model, multi_mode,
                                tp_cfg, w_bundle)
        # model = start_training(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode=multi_mode)
        # First training with both real control modes as covariates continued with the actual boundaries
        # plus the covariates chosen from the max and min val boundaries of the tp_cfg ranges
    return model, tp_cfg, w_bundles, best_rmse


def test_prediction(all_intermediate, all_past_covariates_sim: TimeSeries, bundle, eval_cfg, fc_cfg,
                    model: ForecastingModel | TorchForecastingModel | LightGBMModel | DLinearModel,
                    multi_mode: str | Literal['1.0'], tp_cfg: TrainPredictConfig, w_bundle: WorkerBundle):
    try:
        base_time = bundle.val_scaled.end_time()
        past_series = concatenate([bundle.val_scaled, bundle.test_scaled])
        futures = predict_simulative_control(
            w_bundle.chunks, bundle.val_scaled, fc_cfg, tp_cfg, eval_cfg, model,
            past_series, tp_cfg.actual_boundaries[1], tp_cfg.actual_boundaries[0], base_time,
            simulated_covariates=(all_past_covariates_sim, None), multi=tp_cfg.multi
        )
        evaluate(all_intermediate, futures, fc_cfg, eval_cfg, tp_cfg, w_bundle, mode=multi_mode)
    except Exception as e:
        print(
            f"Error occurred for {fc_cfg} {tp_cfg} {eval_cfg} during debugging {multi_mode}. Continuing...")
        import traceback
        traceback.print_exc()


def start_training(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode):
    if not check_existing_model(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation, tp_cfg.prediction_horizon,
                                input_size=tp_cfg.input_size, serialization_t=get_serial_t(fc_cfg.model_type), mode=mode):
        model = train(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode)
        # convert_and_store_onnx_model(exp, model, fc_cfg.model_type, st_lt, component, segmentation, prediction_horizon, input_size, mode)
    else:
        print(f"CUDA available: {torch.cuda.is_available()} and loading not only weights: {fc_cfg.model_type in [Models.LSTM, Models.DLinear, Models.TCN] and torch.cuda.is_available()}")
        load_kwargs = {"weights_only": False} if fc_cfg.model_type in [Models.LSTM, Models.DLinear, Models.TCN] and torch.cuda.is_available() else {}
        model = get_constructor(fc_cfg.model_type).load(get_simple_model_path(fc_cfg.exp, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation, tp_cfg.prediction_horizon, tp_cfg.input_size, get_serial_t(fc_cfg.model_type), mode=mode), **load_kwargs)
    return model




def train(all_intermediate, bundle, fc_cfg, tp_cfg, parameters, w_bundle, mode, persist=True):
    exp = fc_cfg.exp
    st_lt = fc_cfg.st_lt
    component = tp_cfg.component
    segmentation = tp_cfg.segmentation
    input_size = tp_cfg.input_size
    model_type = fc_cfg.model_type
    prediction_horizon = tp_cfg.prediction_horizon
    prediction_stride = tp_cfg.prediction_stride
    print(f"Starting training for {fc_cfg} {tp_cfg} {mode}...\n")
    if w_bundle.past_covariates is None and "lags_past_covariates" in parameters:
        del parameters["lags_past_covariates"]
    if w_bundle.past_covariates is not None and model_type in [Models.DLinear]:
        parameters["use_static_covariates"] = True
    elif w_bundle.past_covariates is None and model_type in [Models.DLinear]:
        parameters["use_static_covariates"] = False

    train_scaled = bundle.train_scaled
    val_scaled = bundle.val_scaled
    past_covariates = w_bundle.past_covariates
    val_past_covariates = w_bundle.val_past_covariates
    constructor = get_constructor(model_type)
    if tp_cfg.gs:
        # Requires hyperparameter optimization through gridsearch
        if isinstance(train_scaled, list):
            ts_start = datetime.datetime.now()
            # Perform manual gridsearch for components
            model, param, score = gridsearch_component(constructor=constructor,
                                                       param_grid=parameters,
                                                       series=train_scaled,
                                                       val_series=val_scaled,
                                                       past_covariates=past_covariates,
                                                       val_past_covariates=val_past_covariates,
                                                       n_jobs=n_jobs)
        else:
            ts_start = datetime.datetime.now()
            model, param, score = constructor.gridsearch(n_jobs=n_jobs,
                                                         metric=rmse,
                                                         parameters=parameters,
                                                         series=train_scaled,
                                                         val_series=val_scaled,
                                                         past_covariates=past_covariates)
    else:
        ts_start = datetime.datetime.now()
        print(parameters)
        model = constructor(**parameters)

    fit_kwargs = {
        "series": train_scaled,
        "val_series": val_scaled,
    }
    if w_bundle.val_past_covariates is not None:
        fit_kwargs["past_covariates"] = past_covariates
        fit_kwargs["val_past_covariates"] = val_past_covariates
        #fit_kwargs["verbose"] = False
    if model_type in [Models.TCN, Models.DLinear]:
        model.fit(**fit_kwargs)
    else:
        model.fit(**fit_kwargs)
    ts_end = datetime.datetime.now()
    if persist:
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, ts_end, prediction_horizon, prediction_stride,
                            input_size, result_key=end_t,
                            mode=mode)
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, ts_start, prediction_horizon, prediction_stride,
                            input_size, result_key=start_t,
                            mode=mode)
    duration = ts_end - ts_start
    if tp_cfg.gs:
        print(
            f"Finished gridsearch for {exp} {model_type} {st_lt.value} {component.value} {segmentation.value} {mode} with a score of {score} and parameters {param} with training time {duration}...\n")
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, param,
                            prediction_horizon, prediction_stride, input_size,
                            result_key=params_key, mode=mode)
    else:
        print(
            f"Finished training for {exp} {model_type} {st_lt.value} {component.value} {segmentation.value} {mode} with training time {duration}...\n")
    if persist:
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, duration, prediction_horizon, prediction_stride,
                            input_size,
                            result_key=duration_t, mode=mode)
        store_model(exp, model, model_type, st_lt, component, segmentation,  prediction_horizon, input_size, mode=mode)
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component,
                            get_simple_model_path(exp, model_type, st_lt, component, segmentation, prediction_horizon,
                                                  input_size, get_serial_t(fc_cfg.model_type), mode=mode), prediction_horizon, prediction_stride,
                            input_size,
                            result_key=actual_model, mode=mode)
    #with open("pickle.dump", "wb") as f:
    #    pickle.dump(all_intermediate, f)
        store_intermediate(all_intermediate)
    return model



def predict(all_intermediate, bundle, fc_cfg, tp_cfg, w_bundle, model):
    exp = fc_cfg.exp
    st_lt = fc_cfg.st_lt
    component = tp_cfg.component
    segmentation = tp_cfg.segmentation
    input_size = tp_cfg.input_size
    model_type = fc_cfg.model_type
    prediction_horizon = tp_cfg.prediction_horizon
    prediction_stride = tp_cfg.prediction_stride
    val_scaled = get_sensors_only(bundle.val_scaled, fc_cfg)
    print(f"Starting prediction for {fc_cfg} {tp_cfg}...\n")
    ts_start = datetime.datetime.now()
    assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, ts_start, prediction_horizon, prediction_stride, input_size, result_key=start_p)
    past_series = concatenate([val_scaled, get_sensors_only(bundle.test_scaled, fc_cfg)] )
    if model_type is Models.TCN and st_lt is Horizon.ST:
        futures = predict_future_slow(model, bundle, fc_cfg, tp_cfg, w_bundle, past_series)
    else:
        futures = predict_future(model, bundle, tp_cfg, w_bundle, past_series)
    # Currently concatenation is not possible anymore, due to prediction stride

    # pred_future = concatenate(futures)
    ts_end = datetime.datetime.now()
    assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, ts_end, prediction_horizon, prediction_stride, input_size, result_key=end_p)
    assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, ts_end - ts_start, prediction_horizon, prediction_stride, input_size,
                        result_key=duration_p)
    print(f"Ending prediction for {exp} {model_type} {st_lt.value}...\n")
    #display_forecast(pred_future, actual_test_scaled, forecast_type, start_date=actual_test_scaled.start_time(),
    #                 figure_path=get_figure_path(exp, mode_f, model_type, st_lt, component, segmentation))
    store_intermediate(all_intermediate)
    return futures


def evaluate(all_intermediate, futures, fc_cfg, eval_cfg, tp_cfg, w_bundle, mode=None):
    exp = fc_cfg.exp
    st_lt = fc_cfg.st_lt
    component = tp_cfg.component
    segmentation = tp_cfg.segmentation
    input_size = tp_cfg.input_size
    model_type = fc_cfg.model_type
    prediction_horizon = tp_cfg.prediction_horizon
    prediction_stride = tp_cfg.prediction_stride
    chunks = w_bundle.chunks
    for score in Scores:
        #if score is Scores.MASE:
        #    assign_intermediate(all_intermediate, exp, model_type, st_lt,
        #                        metric_assignment[score](actual_test_scaled, pred_future, train_scaled), score_key=score)
        #else:
        # Test if evaluation was already computed for this score
        result = []
        for test, future in zip(chunks, futures):
            result.append(metric_assignment[score](test, future))
        avg = np.average(result)
        print(f"The average {score.value} is {avg} for {get_print_string(exp, model_type, st_lt, component, segmentation, prediction_horizon, prediction_stride, input_size)}")
        assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component,
                                result, prediction_horizon, prediction_stride, input_size, score_key=score.value, mode=mode)
        #else:
        #    result = assign_intermediate(all_intermediate, exp, model_type, st_lt, segmentation, component, "dummy", prediction_horizon, prediction_stride, input_size, score_key=score.value, exist=False, mode=mode)
        #    avg = np.average(result)
        #    print(
        #        f"The average {score.value} is {avg} for {get_print_string(exp, model_type, st_lt, component, segmentation, prediction_horizon, prediction_stride, input_size)}")
    store_intermediate(all_intermediate)


def manual_grid_search(
        constructor,
        param_grid,
        train_series,
        val_series,
        past_covariates,
        val_past_covariates,
        n_jobs=4
):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    configs = [
        dict(zip(keys, v))
        for v in product(*values)
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_config)(
            constructor,
            config,
            train_series,
            val_series,
            past_covariates,
            val_past_covariates
        )
        for config in configs
    )

    return results


def evaluate_config(constructor, config, train_series, val_series, past_covariates, val_past_covariates):
    if past_covariates is None:
        if "lags_past_covariates" in config:
            del config["lags_past_covariates"]
        val_past_covariates = [None] * len(val_series)
    model = constructor(
        **config
    )

    model.fit(series=train_series, past_covariates=past_covariates)

    errors = []
    # RMSE
    i = 0


    for ts_val, ts_cv in zip(val_series, val_past_covariates):
        n = len(ts_val) - config["lags"]
        if n >= 1:
            if i > 0:
                forecast = model.predict(
                    n=n,
                    series=ts_val[:config["lags"]],
                    past_covariates=ts_cv
                )
            else:
                forecast = model.predict(
                    n=n,
                    series=ts_val[:config["lags"]],
                    past_covariates=ts_cv
                )

            i += 1
            errors += [(tr - pred) ** 2 for tr, pred in zip(ts_val[-n:], forecast)]

    return {
        "model": constructor(
            **config
        ),
        "param": config,
        "score": float(math.sqrt(np.mean([err.values()[0][0] for err in errors])))
    }


def historical_vis(model, bundle, fc_cfg, tp_cfg, w_bundle):
    past_series = get_sensors_only(concatenate([bundle.train_scaled, bundle.val_scaled, bundle.test_scaled]), fc_cfg)
    pred_series = model.historical_forecasts(
        past_series,
        start=bundle.val_scaled.start_time(),
        forecast_horizon=tp_cfg.prediction_horizon,
        stride=tp_cfg.prediction_stride,
        past_covariates=w_bundle.past_covariates,
        last_points_only=False,
        retrain=False,
        verbose=True,
    )
    try:
        pred_series = concatenate(pred_series)
        display_forecast(
            pred_series, past_series, f"{fc_cfg.exp} {fc_cfg.model_type} {fc_cfg.st_lt} {tp_cfg}", start_date=bundle.val_scaled.start_time(),
            figure_path=get_figure_path(fc_cfg.exp, mode_h, fc_cfg.model_type, fc_cfg.st_lt, tp_cfg.component, tp_cfg.segmentation, tp_cfg.prediction_stride, tp_cfg.input_size)
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())


def gridsearch_component(constructor, param_grid, series, val_series, past_covariates, n_jobs, val_past_covariates):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    #configs =


    #    dict(zip(keys, v))
    #   for v in product(*values)
    #]
    #res = evaluate_config(constructor, configs[0], series, val_series, past_covariates, val_past_covariates)

    results = manual_grid_search(
        constructor=constructor,
        param_grid=param_grid,
        train_series=series,
        val_series=val_series,
        past_covariates=past_covariates,
        n_jobs=n_jobs,
        val_past_covariates=val_past_covariates
    )

    best_result = min(results, key=lambda x: x["score"])

    return best_result["model"], best_result["param"], best_result["score"]


