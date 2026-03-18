from darts.models import LightGBMModel, DLinearModel, BlockRNNModel, TCNModel
from hdt.parameters import Models


def get_constructor(model_type):
    if model_type is Models.LightGBM:
        return LightGBMModel
    elif model_type is Models.DLinear:
        return DLinearModel
    elif model_type is Models.TCN:
        return TCNModel
    elif model_type is Models.LSTM:
        return BlockRNNModel
