from darts.models import RandomForestModel
from util import *
from experiment.experiment import get_exp_configuration

def get_rf_configuration(exp):
    prediction_horizon, st_input_size, st_horizon, lt_input_size, lt_horizon, sr = get_exp_configuration(exp)
    parameters = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["squared_error"],
        "n_jobs": [1],
        "lags":
            {Horizon.LT: [lt_input_size],
             Horizon.ST: [st_input_size]},
        "lags_past_covariates":
            {Horizon.LT: [lt_input_size],
             Horizon.ST: [st_input_size]},
        "output_chunk_length":
            {Horizon.LT: [lt_horizon],
             Horizon.ST: [st_horizon]},
    }

    configuration_rf = {
        GRIDSEARCH_PARAM: parameters,
        TYPE: Models.RF,
        CONSTRUCTOR: RandomForestModel
    }
    return configuration_rf