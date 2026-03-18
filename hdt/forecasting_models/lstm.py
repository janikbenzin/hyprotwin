# https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/darts/darts_models.py


from darts.models import BlockRNNModel
from hdt.util import *

def extract_horizon(configuration):
    return configuration[PARAM]["output_chunk_length"]

def extract_lags(configuration):
    return configuration[PARAM]["input_chunk_length"]

def extract_lags_past_covariates(configuration):
    return configuration[PARAM]["input_chunk_length"]

def get_lstm_configuration(exp, st_lt):
    configuration_lstm = {
        TYPE: Models.LSTM,
        CONSTRUCTOR: BlockRNNModel,
        LAGS: extract_lags,
        PREDICTION_HORIZON: extract_horizon,
        COV: extract_lags_past_covariates,
        P_GET: get_lstm_param
    }
    return configuration_lstm




def get_lstm_param(input_sizes, horizons, past=True):
    import torch
    #torch.set_num_threads(7)
    if past:
        return {
            'model': 'LSTM',
            #"torch_dtype": torch.float32,
            "input_chunk_length": input_sizes,
            "output_chunk_length": horizons,
        "pl_trainer_kwargs": {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
            "enable_progress_bar": False,
            },
        #"dataloader_kwargs": {"num_worker": 7}
        }
    else:
        return {
            'model': 'LSTM',
            #"torch_dtype": torch.float32,
            "input_chunk_length": input_sizes,
            "output_chunk_length": horizons,
            "pl_trainer_kwargs": {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "enable_progress_bar": False,
            },
            #"dataloader_kwargs": {"num_worker": 7}
        }
