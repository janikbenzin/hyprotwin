import pickle
import warnings

from hdt.forecasting_models.constructor import get_constructor

warnings.filterwarnings("ignore")

from hdt.cleaned_cpee_approach import *
from hdt.forecasting_models.lightgbm import get_gbm_configuration
from hdt.forecasting_models.dlinear import get_dlinear_configuration
from hdt.forecasting_models.tcn import get_tcn_configuration
from hdt.forecasting_models.lstm import get_lstm_configuration
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import multiprocessing as mp
import json
import sys

_GS_ALL_INTERMEDIATE = None
_GS_CHUNKS_TEST = None
_GS_FC_CFG = None
_GS_TP_CFG = None
_GS_EVAL_CFG = None
_GS_BUNDLE = None
_GS_SCALE_MIN = None
_GS_SCALE_MAX = None
_GS_TRAIN_SCALED = None
_GS_VAL_SCALED = None
_GS_TEST_SCALED = None
_GS_RANDOM_STATE = None
_CONTROL_INIT_CACHE = {}



def _make_cache_key(past_series, base_time, input_delta, mode_component):
    return (id(past_series), str(base_time), str(input_delta), mode_component)


def _get_initial_controls_cached(past_series, base_time, input_delta, mode_component):
    cache_key = _make_cache_key(past_series, base_time, input_delta, mode_component)

    if cache_key not in _CONTROL_INIT_CACHE:
        _CONTROL_INIT_CACHE[cache_key] = past_series[base_time - input_delta:base_time][mode_component]

    return _CONTROL_INIT_CACHE[cache_key]


def clear_control_cache():
    global _CONTROL_INIT_CACHE
    _CONTROL_INIT_CACHE.clear()





def train_model_for_boundaries(lower, upper, bundle, fc_cfg, tp_cfg, scale_min, scale_max, random_state):

    train_scaled_sensor = get_sensors_only(bundle.train_scaled, fc_cfg)
    val_scaled_sensor = get_sensors_only(bundle.val_scaled, fc_cfg)

    # Build ts_sim
    mode_component = get_all_mode_names(fc_cfg.exp)
    #last_control = w_bundles[0].past_covariates[get_all_mode_names(fc_cfg.exp)].first_value()

    sim_mode_component = get_all_sim_mode_components(fc_cfg.exp)
    ts_sim = build_ts_sim_from_covariates(bundle, bundle.train_scaled[mode_component].first_value(), lower, upper, fc_cfg,
                                                       scale_min, scale_max, sim_mode_component)
    past_covariates_sim = ts_sim.drop_after(val_scaled_sensor.start_time())
    val_past_covariates_sim = ts_sim.drop_before(train_scaled_sensor.end_time()).drop_after(val_scaled_sensor.end_time(), keep_point=True)
    if tp_cfg.multi:
        print(f"{tp_cfg.multi} stacking covariates...", flush=True)
        ts_sim, past_covariates_sim, val_past_covariates_sim = stack_covariates_for_multi(
            ts_sim, bundle, mode_component, past_covariates_sim, val_past_covariates_sim)
    model_type = fc_cfg.model_type
    st_lt = fc_cfg.st_lt
    exp = fc_cfg.exp

    # Create configurations
    if model_type is Models.LightGBM:
        configuration = get_gbm_configuration(exp, st_lt)
    elif model_type is Models.DLinear:
        configuration = get_dlinear_configuration(exp, st_lt)
    elif model_type is Models.TCN:
        configuration = get_tcn_configuration(exp, st_lt)
    else:
        configuration = get_lstm_configuration(exp, st_lt)
    parameters = configuration[P_GET](tp_cfg.input_size, 1, tp_cfg.component is Components.ALL)

    parameters["random_state"] = random_state
    if model_type is Models.LightGBM:
        parameters["num_threads"] = 3

    constructor = get_constructor(fc_cfg.model_type)


    file_name = get_model_file_string(lower, upper, exp, tp_cfg.fold, model_type, tp_cfg.prediction_horizon, tp_cfg.prediction_stride, tp_cfg.input_size, tp_cfg.multi)
    load_kwargs = {"weights_only": False} if model_type in [Models.LSTM, Models.DLinear, Models.TCN] and torch.cuda.is_available() else {}
    #load_kwargs = {}
    if os.path.exists(file_name):
        model = constructor.load(file_name, **load_kwargs)
        print(f"Loaded prediction model from {file_name}", flush=True)
    else:
        model = constructor(**parameters)
        model.fit(
        series=train_scaled_sensor.astype(np.float32), #if model_type in [Models.DLinear, Models.LSTM] else train_scaled_sensor,
        val_series=val_scaled_sensor.astype(np.float32), # if model_type in [Models.DLinear, Models.LSTM] else val_scaled_sensor,
        past_covariates=past_covariates_sim.astype(np.float32), #if model_type in [Models.DLinear, Models.LSTM] else past_covariates_sim,
        val_past_covariates=val_past_covariates_sim.astype(np.float32)# if model_type in [Models.DLinear, Models.LSTM] else val_past_covariates_sim
    )
        model.save(file_name)
        print(f"Trained and saved prediction model to {file_name}", flush=True)

    return model, ts_sim.astype(np.float32) #if model_type in [Models.DLinear, Models.LSTM] else ts_sim)


def predict_with_simulated_control(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model,
                                   all_past_covariates):
    """Predict using the trained model with simulated control."""
    base_time = bundle.val_scaled.end_time()
    return predict_simulative_control(
        chunks_test, bundle.val_scaled, fc_cfg, tp_cfg, eval_cfg, model,
        concatenate([bundle.val_scaled, bundle.test_scaled]), upper, lower, base_time,
        _get_initial_controls_cached, simulated_covariates=all_past_covariates, multi=tp_cfg.multi
    )


def predict_with_simulated_control_val(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model,
                                       all_past_covariates):
    """Predict on validation set using the trained model with simulated control."""
    base_time = bundle.train_scaled.end_time()
    return predict_simulative_control(
        chunks_test, bundle.train_scaled, fc_cfg, tp_cfg, eval_cfg, model,
        concatenate([bundle.train_scaled, bundle.val_scaled]), upper, lower, base_time,
        _get_initial_controls_cached, simulated_covariates=all_past_covariates, multi=tp_cfg.multi
    )


def gridsearch_boundaries_val(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model, all_past_covariates):
    """Evaluate on validation set."""
    test_futures = predict_with_simulated_control_val(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg,
                                                      model, all_past_covariates)
    avgs = quick_evaluation(chunks_test, test_futures, fc_cfg, tp_cfg)
    print(f"Validating with {lower} and {upper} done: {avgs[1]}.", flush=True)
    return avgs[1]


def gridsearch_boundaries_all(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model, all_past_covariates):
    """Evaluate on test set."""
    test_futures = predict_with_simulated_control(chunks_test, bundle, fc_cfg, tp_cfg, lower, upper, eval_cfg, model,
                                                  all_past_covariates)
    avgs = quick_evaluation(chunks_test, test_futures, fc_cfg, tp_cfg)
    print(f"Testing with {lower} and {upper} done: {avgs[1]}.", flush=True)
    return avgs[1]




def _gridsearch_worker_all_fork(lower: float, upper: float):
    """Worker function for test set evaluation. Trains model inside worker."""
    try:
        # Train model specific to these boundaries
        model, ts_sim = train_model_for_boundaries(
            lower, upper, _GS_BUNDLE, _GS_FC_CFG, _GS_TP_CFG,
            _GS_SCALE_MIN, _GS_SCALE_MAX, _GS_RANDOM_STATE
        )

        # Evaluate
        metric = gridsearch_boundaries_all(
            chunks_test=_GS_CHUNKS_TEST,
            bundle=_GS_BUNDLE,
            fc_cfg=_GS_FC_CFG,
            tp_cfg=_GS_TP_CFG,
            lower=lower,
            upper=upper,
            eval_cfg=_GS_EVAL_CFG,
            model=model,
            all_past_covariates=(ts_sim, None)
        )

        return (lower, upper), metric
    except Exception as e:
        print(f"Error in worker for l={lower}, u={upper}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (lower, upper), float('inf')


def _gridsearch_worker_val_fork(lower: float, upper: float):
    try:
        # Train model specific to these boundaries
        model, ts_sim = train_model_for_boundaries(
            lower, upper, _GS_BUNDLE, _GS_FC_CFG, _GS_TP_CFG,
            _GS_SCALE_MIN, _GS_SCALE_MAX, _GS_RANDOM_STATE
        )

        # Evaluate
        metric = gridsearch_boundaries_val(
            chunks_test=_GS_CHUNKS_TEST,
            bundle=_GS_BUNDLE,
            fc_cfg=_GS_FC_CFG,
            tp_cfg=_GS_TP_CFG,
            lower=lower,
            upper=upper,
            eval_cfg=_GS_EVAL_CFG,
            model=model,
            all_past_covariates=(ts_sim, None)
        )

        return (lower, upper), metric
    except Exception as e:
        print(f"Error in worker for l={lower}, u={upper}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (lower, upper), float('inf')


def _parallel_gridsearch_processes_fork(pairs, worker_fn, max_workers: int | None = None):
    ctx = mp.get_context("fork")
    results = {}
    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
    ) as ex:
        futures = {ex.submit(worker_fn, lo, up): (lo, up) for (lo, up) in pairs}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            (lo, up), metric = fut.result()
            results[(lo, up)] = metric
            done += 1
            if done % 10 == 0 or done == total:
                print(f"Gridsearch progress: {done}/{total} | Best so far: {min(results.values()):.4f}", flush=True)
    return results

def decode_cfg(encoded_str):
    return base64.b64decode(encoded_str).decode('utf-8')

def main(fc, tp, ev):
    logging.disable(logging.CRITICAL)

    fc_cfg = ForecastConfig.from_json(decode_cfg(fc))
    tp_cfg = TrainPredictConfig.from_json(decode_cfg(tp))
    eval_cfg = EvalConfig.from_json(decode_cfg(ev))

    prediction_stride = tp_cfg.prediction_stride
    prediction_horizon = tp_cfg.prediction_horizon
    input_size = tp_cfg.input_size

    # Configuration
    exp = fc_cfg.exp  # Default experiment
    st_lt = fc_cfg.st_lt
    model_type = fc_cfg.model_type
    component = tp_cfg.component
    fold = tp_cfg.fold
    bundle_file = tp_cfg.bundle_file

    # Load data
    all_intermediate = load_latest_intermediate()
    if not all_intermediate:
        all_intermediate = {exp: get_intermediate(exp)}
    elif exp not in all_intermediate:
        all_intermediate[exp] = get_intermediate(exp)

    with open(bundle_file, "rb") as f:
        bundle, scale_min, scale_max = pickle.load(f)



    # Set globals for workers (no longer include model or ts_sim)
    global _GS_ALL_INTERMEDIATE, _GS_BUNDLE, _GS_CHUNKS_TEST, _GS_FC_CFG, _GS_TP_CFG, _GS_EVAL_CFG
    global _GS_SCALE_MIN, _GS_SCALE_MAX, _GS_TRAIN_SCALED, _GS_VAL_SCALED, _GS_TEST_SCALED, _GS_RANDOM_STATE

    _GS_ALL_INTERMEDIATE = all_intermediate
    _GS_BUNDLE = bundle
    _GS_FC_CFG = fc_cfg
    _GS_TP_CFG = tp_cfg
    _GS_EVAL_CFG = eval_cfg
    _GS_SCALE_MIN = scale_min
    _GS_SCALE_MAX = scale_max
    _GS_TRAIN_SCALED = bundle.train_scaled.astype(np.float32) #if model_type in [Models.DLinear, Models.LSTM] else bundle.train_scaled
    _GS_VAL_SCALED = bundle.val_scaled.astype(np.float32) #if model_type in [Models.DLinear, Models.LSTM] else bundle.val_scaled
    _GS_TEST_SCALED = bundle.test_scaled.astype(np.float32) #if model_type in [Models.DLinear, Models.LSTM] else bundle.test_scaled
    _GS_RANDOM_STATE = random_state

    print("Starting gridsearch for simulated control modes...", flush=True)

    # Create chunks on validation set

    chunk_duration = pd.Timedelta(prediction_horizon * SAMPLING_FACTOR, "s")
    stride_duration = pd.Timedelta(prediction_stride * SAMPLING_FACTOR, "s")
    chunks_val = chunk_series(bundle.val_scaled, chunk_duration, stride_duration)

    _GS_CHUNKS_TEST = chunks_val

    max_vals_coarse = tp_cfg.upper_vals
    min_vals_coarse = tp_cfg.lower_vals
    pairs_coarse = list(product(min_vals_coarse, max_vals_coarse))

    print(f"Total coarse configurations to evaluate: {len(pairs_coarse)}", flush=True)
    print(f"Lower bounds: {min_vals_coarse}", flush=True)
    print(f"Upper bounds: {max_vals_coarse}", flush=True)

    ts_start = datetime.datetime.now()



    max_worker = WORKERS_DEFAULT if fc_cfg.exp != SWAT else 1
    if model_type in [Models.LightGBM]:
        max_worker = WORKERS_GBM
    elif model_type is Models.DLinear:
        max_worker = WORKERS_DLINEAR
    combinations_rmse_val = _parallel_gridsearch_processes_fork(
        pairs_coarse,
        worker_fn=_gridsearch_worker_val_fork,
        max_workers=max_worker,
    )
    ts_end = datetime.datetime.now()
    print(f"Start: {ts_start}", flush=True)
    print(f"End: {ts_end}", flush=True)
    print(f"Runtime: {ts_end - ts_start}", flush=True)
    try:
        with open(get_gridsearch_file_string(exp, fold, model_type, prediction_horizon, prediction_stride, input_size), "w") as f:
            json.dump({str(k): v for k, v in combinations_rmse_val.items()}, f, indent=2, default=custom_json_serializer)
    except Exception:
        print(f"Error writing detailed results to disk {fc_cfg} {tp_cfg}")
    best_lower, best_upper = min(combinations_rmse_val, key=combinations_rmse_val.get)
    best_rmse = combinations_rmse_val[(best_lower, best_upper)]

    print(f"Best upper lower")
    print(f"Lower bound: {best_lower}", flush=True)
    print(f"Upper bound: {best_upper}", flush=True)
    print(f"Validation RMSE: {best_rmse}", flush=True)
    clear_control_cache()



if __name__ == "__main__":

        # Example: python parallelized_sim_boundaries.py --skip-coarse 79.2 99.4
    if len(sys.argv) != 4:
        print("Usage: fc tp eval")
        sys.exit(1)
    fc = sys.argv[1]
    tp = sys.argv[2]
    ev = sys.argv[3]
    print(f"Running with {fc} {tp} {ev}", flush=True)
    main(fc, tp, ev)
