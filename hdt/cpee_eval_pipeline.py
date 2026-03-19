from hdt.forecasting_models.lightgbm import get_gbm_configuration
from hdt.forecasting_models.dlinear import get_dlinear_configuration
from hdt.forecasting_models.tcn import get_tcn_configuration
from hdt.forecasting_models.lstm import get_lstm_configuration
from hdt.run.run import configured_train_predict_evaluate
import warnings
import logging
from hdt.parameters import (
    Models,
    RUN_MODELS,
    RUN_EXPERIMENTS,
    RUN_HORIZONS,
    EXPERIMENTS,
    TS_CV_FOLDS,
    ASSIGNED_FOLDS,
    target_workers,
    ALLOW_SIMULATION,
    ADD_EVAL_VALUES_TO_FINAL_RESULTS
)
from hdt.util import load_latest_intermediate, get_intermediate
from hdt.run.hdt_evaluation_centralized import run_predictive_simulation_and_extract_parallel
import sys
import os



sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def assign_model_and_run(model, exp, st_lt):
    if model in RUN_MODELS:
        if model is Models.LightGBM:
            return configured_train_predict_evaluate(get_gbm_configuration(exp, st_lt))
        elif model is Models.DLinear:
            return configured_train_predict_evaluate(get_dlinear_configuration(exp, st_lt))
        elif model is Models.TCN:
            return configured_train_predict_evaluate(get_tcn_configuration(exp, st_lt))
        elif model is Models.LSTM:
            return configured_train_predict_evaluate(get_lstm_configuration(exp, st_lt))
        else:
            return None
    else:
        return None


logging.disable(logging.CRITICAL)

if ADD_EVAL_VALUES_TO_FINAL_RESULTS:
    #all_intermediate = load_latest_intermediate(pattern=r"final_results_dict_2026-03-14 12:01:55.540004.json")
    all_intermediate = load_latest_intermediate(pattern=r"final_results_dict_(.+)\.json")
else:
    all_intermediate = load_latest_intermediate()
if not all_intermediate:
    all_intermediate = {i: get_intermediate(i) for i in EXPERIMENTS}

for exp in RUN_EXPERIMENTS:
    if exp not in all_intermediate:
        all_intermediate[exp] = get_intermediate(exp)

    # Collect all model jobs first
    model_jobs = []
    for model in RUN_MODELS:
        for st_lt in RUN_HORIZONS:
            model_assignment = assign_model_and_run(model, exp, st_lt)
            if model_assignment is not None:
                if ASSIGNED_FOLDS is None:
                    for fold in range(TS_CV_FOLDS):
                        model_jobs.append((model, exp, st_lt, fold, model_assignment))
                else:
                    model_jobs.append((model, exp, st_lt, ASSIGNED_FOLDS, model_assignment))

    # Process models in parallel using worker pool
    run_predictive_simulation_and_extract_parallel(model_jobs, all_intermediate, target_workers, allow_simulation=ALLOW_SIMULATION)