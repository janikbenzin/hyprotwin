from darts.models import DLinearModel
from hdt.util import *


def extract_horizon(configuration):
    return configuration[PARAM]["output_chunk_length"]

def extract_lags(configuration):
    return configuration[PARAM]["input_chunk_length"]

#def extract_lags_past_covariates(configuration):
#    return configuration[PARAM]["lags_past_covariates"]

def get_dlinear_configuration(exp, st_lt):
    configuration_dlinear = {
        TYPE: Models.DLinear,
        CONSTRUCTOR: DLinearModel,
        LAGS: extract_lags,
        PREDICTION_HORIZON: extract_horizon,
        #COV: extract_lags_past_covariates,
        P_GET: get_dlinear_param
    }
    return configuration_dlinear




def get_dlinear_param(input_sizes, horizons, past=True):
    import torch    # Apparently, the lags for the past_covariates are not needed as in other models from Darts
    # https://github.com/GestaltCogTeam/BasicTS/blob/v0.5.8/baselines/DLinear/ETTm2.py
    # https://arxiv.org/pdf/2205.13504
    return {
        "input_chunk_length": input_sizes,
        "output_chunk_length": horizons,
        "batch_size": 64,
        "shared_weights": True,
        "work_dir": "./models_tmp/",
        "loss_fn": torch.nn.L1Loss(),
        "optimizer_kwargs": {
            "lr": 0.0003,
            "weight_decay": 0.0001,
        },
        "lr_scheduler_cls": torch.optim.lr_scheduler.MultiStepLR,
        "lr_scheduler_kwargs" : {
            "milestones": [1, 25],
            "gamma": 0.5
        },
        "pl_trainer_kwargs" : {
            "gradient_clip_val": 5.0,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "enable_progress_bar": False,
        }
        }