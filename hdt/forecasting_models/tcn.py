from hdt.util import *
from darts.models import TCNModel
import torch

def extract_horizon(configuration):
    return configuration[PARAM]["output_chunk_length"]

def extract_lags(configuration):
    return configuration[PARAM]["input_chunk_length"]

def get_tcn_configuration(exp, st_lt):
    configuration_tcn = {
        TYPE: Models.TCN,
        CONSTRUCTOR: TCNModel,
        LAGS: extract_lags,
        PREDICTION_HORIZON: extract_horizon,
        P_GET: get_tcn_param
    }
    return configuration_tcn

def get_tcn_param(input_sizes, horizons, past=True):
    # https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/moderntcn/moderntcn.py
    import torch
    torch.set_num_threads(7)
    return {
        "input_chunk_length": input_sizes,
        "output_chunk_length": horizons,
        "n_epochs": 100,
        "batch_size": 128,
        "optimizer_kwargs": {"lr": 0.0001},
        "kernel_size": 25,  # Darts TCN default/common
        "num_filters": 256,
        "weight_norm": True,
        "dilation_base": 2,
        "dropout": 0.1,
        "pl_trainer_kwargs": {
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "enable_progress_bar": False,
        }
    }