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
from darts.models import ExponentialSmoothing, TransformerModel
from darts.utils.callbacks import TFMProgressBar

from util import chunk_series


def fit_informer(exp,  all_intermediate, series_scaled, train_scaled, val_scaled, test_scaled, chunks, st_input_size, st_horizon, lt_input_size, lt_horizon, sr):
    return


chunk_duration = pd.Timedelta(minutes=5)
chunks = chunk_series(test_scaled, chunk_duration)



model_name = "informer_heating_base"
model_informer = TransformerModel(
    input_chunk_length=12,
    output_chunk_length=1,
    batch_size=32,
    n_epochs=200,
    model_name=model_name,
    nr_epochs_val_period=10,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    random_state=42,
    save_checkpoints=True,
    force_reset=True,
)

model_informer .fit(train_scaled, val_series=val_scaled)
model_informer = TransformerModel.load_from_checkpoint(model_name=model_name, best=True)

pred_series = model_informer.historical_forecasts(
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
        futures.append(model_informer.predict(n=300, series=val_scaled))
    else:
        futures.append(model_informer.predict(n=300, series=chunks[chunk - 1]))



pred_future = concatenate(futures)
display_forecast(pred_future, concatenate(chunks), "1s", start_date=test_scaled.start_time())
plt.show()



model_name = "informer_heating_complete"
model_informer_complete = TransformerModel(
    input_chunk_length=900,
    output_chunk_length=300,
    batch_size=32,
    n_epochs=200,
    model_name=model_name,
    nr_epochs_val_period=10,
    d_model=16,
    nhead=8,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    random_state=42,
    save_checkpoints=True,
    force_reset=True,
)

model_informer_complete.fit(train_scaled, val_series=val_scaled)
model_informer_complete = TransformerModel.load_from_checkpoint(model_name=model_name, best=True)



futures = []
for chunk, chunk_data in enumerate(chunks):
    if chunk == 0:
        futures.append(model_informer_complete.predict(n=300, series=val_scaled))
    else:
        futures.append(model_informer_complete.predict(n=300, series=concatenate([val_scaled] + chunks[:chunk ])))



pred_future = concatenate(futures)
display_forecast(pred_future, concatenate(chunks), "1s", start_date=test_scaled.start_time())
plt.show()
