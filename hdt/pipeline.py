from hdt.forecasting_models.lightgbm import get_gbm_configuration
from hdt.forecasting_models.dlinear import get_dlinear_configuration
from hdt.forecasting_models.tcn import get_tcn_configuration
from hdt.forecasting_models.lstm import get_lstm_configuration
from hdt.run.run import configured_train_predict_evaluate
from hdt.util import *
import warnings

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


all_intermediate = load_latest_intermediate()
if not all_intermediate:
    all_intermediate = {i: get_intermediate(i) for i in EXPERIMENTS}


for exp in RUN_EXPERIMENTS:
    if exp not in all_intermediate:
        all_intermediate[exp] = get_intermediate(exp)
    for model in Models:
        for st_lt in RUN_HORIZONS:
            model_assignment = assign_model_and_run(model, exp, st_lt)
            if model_assignment is not None:
                if ASSIGNED_FOLDS is None:
                    for fold in range(TS_CV_FOLDS):
                        model_assignment(exp, st_lt, all_intermediate, fold)
                else:
                    model_assignment(exp, st_lt, all_intermediate, ASSIGNED_FOLDS)

