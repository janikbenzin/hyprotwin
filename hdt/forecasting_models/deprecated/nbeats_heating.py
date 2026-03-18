import json
import pandas as pd


#df_lin.to_csv("heating_lin.csv")
#df_lin.to_csv("data/heating_2h/heating_lin.csv")


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from darts.datasets import EnergyDataset
from darts.metrics import r2_score
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar


import logging

logging.disable(logging.CRITICAL)

### NBEATS EX


model_name = "nbeats_interpretable_run"
model_nbeats = NBEATSModel(
    input_chunk_length=12,
    output_chunk_length=1,
    generic_architecture=False,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=800,
    random_state=42,
    model_name=model_name,
    save_checkpoints=True,
    force_reset=True,
    **generate_torch_kwargs(),
)

model_nbeats.fit(train_scaled, val_series=val_scaled)
model_nbeats = NBEATSModel.load_from_checkpoint(model_name=model_name, best=True)

pred_series = model_nbeats.historical_forecasts(
    series_scaled,
    start=val.start_time(),
    forecast_horizon=1,
    stride=1,
    last_points_only=False,
    retrain=False,
    verbose=True,
)
pred_series = concatenate(pred_series)
display_forecast(
    pred_series, series_scaled, "1s", start_date=val_scaled.start_time()
)
plt.show()


#
futures = []
for chunk, chunk_data in enumerate(chunks):
    if chunk == 0:
        futures.append(model_nbeats.predict(n=300, series=val_scaled))
    else:
        futures.append(model_nbeats.predict(n=300, series=chunks[chunk - 1]))



pred_future = concatenate(futures)
display_forecast(pred_future, concatenate(chunks), "1s", start_date=test_scaled.start_time())
plt.show()



model_name = "nbeats_interpretable_run_complete"
model_nbeats_complete = NBEATSModel(
    input_chunk_length=900,
    output_chunk_length=300,
    generic_architecture=False,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=800,
    random_state=42,
    model_name=model_name,
    save_checkpoints=True,
    force_reset=True,
    **generate_torch_kwargs(),
)

model_nbeats_complete.fit(train_scaled, val_series=val_scaled)
model_nbeats_complete = NBEATSModel.load_from_checkpoint(model_name=model_name, best=True)



futures = []
for chunk, chunk_data in enumerate(chunks):
    if chunk == 0:
        futures.append(model_nbeats_complete.predict(n=300, series=val_scaled))
    else:
        futures.append(model_nbeats_complete.predict(n=300, series=concatenate([val_scaled] + chunks[:chunk ])))



pred_future = concatenate(futures)
display_forecast(pred_future, concatenate(chunks), "1s", start_date=test_scaled.start_time())
plt.show()
