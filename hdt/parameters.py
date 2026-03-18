import logging
import os
from enum import Enum
import yaml
from pathlib import Path
from darts.metrics import rmse, mae, wmape

logging.disable(logging.CRITICAL)


class Models(Enum):
    LightGBM = "LightGBM"
    DLinear = "DLinear"
    TCN = "FEDFormer" # is actually a TCN despite its value, see forecasting_models
    LSTM = "exponential" # is actually a LSTM despite its value

# Determines whether HyProTwin as based on CPEE is skipped or not, if skipped, then all forecasting models are only trained
SKIP_CPEE = True
# Determines whether the direct forecasting models should be evaluated
ALWAYS_EVAL = True
# Determines whether evaluation should be done one the validation or test dataset
EVAL_ON_VAL = True
# Determines for HyProTwin models, whether the standard and/or (opt) models should be evaluated on validation data set for deciding what model is better
RUN_REAL_SIMULATED = [True, True]
# Predictive simulation scenarios turned on
RUN_SCENARIO = True
# Determines for the HyProTwin, whether the CPEE runs with forecasting models from HyProTwin vs. HyProTwin (opt)
ALLOW_SIMULATION = False

SAMPLING_RATE = "1s"
SAMPLING_FACTOR = int(SAMPLING_RATE[0])
HEATING_2H = "heating_2h"
HEATING = "heating_12h"
WATERLEVEL = "waterlevel_12h"
IRRIGATION = "irrigation_1y"
#COTTON = "cotton_candy"
SWAT = "swat_a1a2"
SCALE = "scale"
SCALE_MIN = "scale_min"
SCALE_MAX = "scale_max"

# All models in the pipeline
RUN_MODELS = [m for m in Models]
RUN_EXPERIMENTS = [HEATING,WATERLEVEL,IRRIGATION]
WORKERS_DEFAULT = 2
WORKERS_GBM = 1
WORKERS_DLINEAR = 1
RECOMPUTE_FUTURES = True
ADD_EVAL_VALUES_TO_FINAL_RESULTS = False


class Horizon(Enum):
    ST = "short-term"  # these are one step predicting forecasting models (F=1)
    LT = "long-term" # these are direct forecasting models (e.g., F=300)

class Scores(Enum):
    WAPE = "WAPE"
    RMSE = "RMSE"
    MAE = "MAE"

class Components(Enum):
    ALL = "all"   # remnant of older version, all forecasting models are trained with Components.ALL
    SENSORS_ONLY = "sensors"

class Segmentation(Enum):
    ON = "on"  # means both simulated and real covariates, mode=1.0 both, mode=0.0, only the simulated without real modes
    OFF = "off" # means only the real are further simulated


RUN_HORIZONS = [Horizon.ST, Horizon.LT]
RUN_COMPONENTS = [Components.ALL]
RUN_PREDICTION_HORIZONS = {HEATING: [300, 1800],
                           WATERLEVEL: [300, 1800],
                           IRRIGATION: [24, 72],
                           #COTTON: [],
                           SWAT: [300, 1800]}
INITIAL_MODE_VALUE = {HEATING: "Off",
                      WATERLEVEL: "Off",
                      IRRIGATION: "Irrigation Off"}


RUN_INPUT_SIZES =  {
                           HEATING: {300: [300, 900, 1800],
                                     1800: [1800]},  # maybe 5400
                           WATERLEVEL: {300: [300, 900, 1800],
                                     1800: [1800]},  # maybe 5400
                           IRRIGATION: {24: [24, 48, 72],
                                     72: [72]},  # maybe 216
                           SWAT: {1800: [1800], # maybe 5400
                                       300: [900]}}




COARSE_BOUNDARIES = {
    HEATING: {
        "min_vals": [76 + i * 1 for i in range(7)],
        "max_vals": [96 + i * 1 for i in range(5)]
    },
    WATERLEVEL: {
        "min_vals": [7 + i * 1 for i in range(7)],
        "max_vals": [27 + i * 1 for i in range(7)]},
    IRRIGATION: {
        "min_vals": [0.18 + i * 0.01 for i in range(7)],
        "max_vals": [0.37 + i * 0.01 for i in range(7)]
    },
    SWAT: {
        "min_vals": [],
        "max_vals": []
    }
}



def load_config(config_path: str) -> dict:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(path, 'r') as file:
        try:
            # Safe_load prevents execution of arbitrary code in the YAML
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            return {}



config_data = load_config('hdt/config.yaml')

    # Accessing your specific key
target_server = config_data.get('target_input_path')
target_user = config_data.get('target_user')
target_host = config_data.get('target_host')
target_start = config_data.get('target_start')
if "target_cpee" in config_data:
    target_cpee = config_data.get('target_cpee')
if "worker_urls" in config_data:
    target_workers = config_data.get('worker_urls')

CPEE_INSTANCE = "CPEE-INSTANCE"
CPEE_INSTANCE_URL = "CPEE-INSTANCE-URL"


EXTRACTOR_TASK = "Read"
SIM_EXTRACTOR_TASKS = {
    HEATING: ["On", "Off"],
    WATERLEVEL:  ["On", "Off"],
    IRRIGATION: ["On", "Off"],
    SWAT: []
}

# No gridsearch for the real system, as too computationally intensive for a composed system in which we have no model of individual components
# as the only composed model we have is discovered from the data (Expert systems with applications), so explosion due to multiplication
MAX_VALS = lambda best_upper: [best_upper - 0.1 + i * 0.05 for i in range(5)]
MIN_VALS = lambda best_lower: [best_lower - 0.1 + i * 0.05 for i in range(5)]
SCENARIO_TIMES = {
    HEATING: 1800,
    WATERLEVEL: 300,
    IRRIGATION: 168
}

DEBUG_RUN = False

MODES = {HEATING: ["heat"],
         WATERLEVEL: ["flow"],
         IRRIGATION: ["irrigate"],
         SWAT: ['MV101', 'P101', 'MV201', 'P201', 'P203', 'P205', 'MV301', 'MV302', 'MV303', 'MV304', 'P302', 'P402', 'P403', 'UV401', 'P501', 'P602']}
EXPERIMENTS = [HEATING, WATERLEVEL, IRRIGATION, SWAT]

PREPROCESSING_INST = True
DIFFERENCE_TS = "t_since_obs"
TIME = "time"
TEMP = "temperature"
LEVEL = "level"
MOIST = "moisture"
SENSOR_PREFIX = "sensor_"
MODE_PREFIX = "mode_"
P_GET = "get"

SENSOR_NAMES = {
    HEATING: [SENSOR_PREFIX + TEMP],
    WATERLEVEL: [SENSOR_PREFIX + LEVEL],
    IRRIGATION: [SENSOR_PREFIX + MOIST]
}

PATH_PREFIX = "/".join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
MODEL_PREFIX = "darts_logs"
random_state: int = 7231
n_jobs = -1
stride = 1
mode_h = "historic"
mode_f = "future"

TYPE = "model_type"
GRIDSEARCH_PARAM = "gridsearch"
PARAM = "parameters"
CONSTRUCTOR = "constructor"
LAGS = "lags"
PREDICTION_HORIZON = "p_horizon"
COV = "covariates"

count = 0
variables_all = {exp: {} for exp in EXPERIMENTS}

start_t = "start_training"
end_t = "end_training"
duration_t = "duration_training"
start_p = "start_prediction"
end_p = "end_prediction"
duration_p = "duration_prediction"
params_key = "model_parameters"
actual_model = "actual_model"
scores = "scores"

FINAL_MODEL_NAMES = {
    "LightGBM": "LightGBM",
    "DLinear": "DLinear",
    "FEDFormer": "TCN", # the enum key is true, the value is deprecated,
    "LSTM": "LSTM"
}

ACTUAL_BOUNDARIES = {
    HEATING: [80,100],
    WATERLEVEL: [10, 30],
    IRRIGATION: [0.21, 0.4],
    SWAT: []
}

TS_CV_FOLDS = 1 # 3
ASSIGNED_FOLDS = 0


# = {HEATING: [6, 9, 60],
#                      WATERLEVEL: [6, 9, 18],
#                      IRRIGATION: [6, 9, 18],
#                      COTTON: [],
#                      SWAT: []}

RUN_PREDICTION_STRIDES = {HEATING: [15, 30], # [15, 30]
                          WATERLEVEL: [15, 30],
                          IRRIGATION: [12, 24],
                          SWAT: [15, 30]} # the process is slow moving

SKIP_LONG_STRIDES = True

metric_assignment = {
    Scores.WAPE: wmape,
    Scores.RMSE: rmse,
    Scores.MAE: mae,
    #Scores.MASE: mase,
    #Scores.SMAPE: smape,
    #Scores.OWA: lambda actual, pred: (smape(actual, pred) + mase(actual, pred)) / 2,
}

modes_assignment = {
    HEATING: ["1.0", "0.0"],
    WATERLEVEL: ["1.0", "0.0"],
    IRRIGATION: ["1.0", "0.0"],
    #COTTON: ["1.0", "0.0"],
    SWAT: ["1.0", "0.0"]
}

def get_intermediate(exp):
    return {model.value:
                {Segmentation.OFF.value:
                     {component.value:
                        {st_lt.value:
                             get_results_dict(exp)
                            for st_lt in Horizon
                        } for component in Components
                     },
                Segmentation.ON.value:
                    {component.value:
                         {
                             mode: get_results_dict(exp) for mode in modes_assignment[exp]
                         }
                        for component in Components
                     }
                 } for model in Models
            }

# Three evaluation parameters are varied: Prediction horizon, input_size (lt_horizon = prediction horizon), prediction_stride
def get_results_dict(exp):
    return {str(prediction_horizon):
                {str(st_input_size):
                     {str(prediction_stride):
                        {start_t: None,
                                end_t: None,
                                duration_t: None,
                                start_p: None,
                                end_p: None,
                                duration_p: None,
                                actual_model: None,
                                params_key: None,
                                scores: {
                                      # Will contain a list with the scores per window
                                         score: None for score in Scores
                                        }
                                }
                      for prediction_stride in RUN_PREDICTION_STRIDES[exp]}
                 for st_input_size in RUN_INPUT_SIZES[exp][prediction_horizon]}
            for prediction_horizon in RUN_PREDICTION_HORIZONS[exp]
    }
