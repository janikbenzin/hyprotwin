from darts.models import LightGBMModel
from hdt.util import *

def extract_horizon(configuration):
    return configuration[PARAM]["output_chunk_length"]

def extract_lags(configuration):
    return configuration[PARAM]["lags"]

def extract_lags_past_covariates(configuration):
    return configuration[PARAM]["lags_past_covariates"]

def get_gbm_configuration(exp, st_lt):
    configuration_gbm = {
        TYPE: Models.LightGBM,
        CONSTRUCTOR: LightGBMModel,
        LAGS: extract_lags,
        PREDICTION_HORIZON: extract_horizon,
        COV: extract_lags_past_covariates,
        P_GET: get_gbm_param
    }
    return configuration_gbm




def get_gbm_param(input_sizes, horizons, past=True):
    if past:
        return {
            "lags": input_sizes,
            "output_chunk_length": horizons,
            "lags_past_covariates": input_sizes
        }
    else:
        return {
            "lags": input_sizes,
            "output_chunk_length": horizons
        }