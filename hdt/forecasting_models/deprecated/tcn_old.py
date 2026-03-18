
from hdt.util import *

import pandas as pd
import torch
from neuralforecast.models import TCN
from neuralforecast import NeuralForecast
from darts import TimeSeries

if not hasattr(pd.Series, 'is_nan'):
    pd.Series.is_nan = lambda self: self.isna()
if not hasattr(pd.DataFrame, 'is_nan'):
    pd.DataFrame.is_nan = lambda self: self.isna()

if not hasattr(pd.Series, 'is_null'):
    pd.Series.is_null = lambda self: self.isnull()
if not hasattr(pd.DataFrame, 'is_null'):
    pd.DataFrame.is_null = lambda self: self.isnull()



class NixtlaTCN():
    def __init__(
            self,
            input_chunk_length: int,
            output_chunk_length: int,
            n_epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            add_encoders: dict = None,
            **tcn_kwargs
    ):
        #super().__init__(add_encoders=add_encoders)
        self.add_encoders = add_encoders
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        self.tcn_kwargs = tcn_kwargs
        self.model = None
        self.training_series = None


    @property
    def input_chunk_length(self) -> int:
        return self._input_chunk_length

    @property
    def output_chunk_length(self) -> int:
        return self._output_chunk_length

    @property
    def _model_encoder_settings(self):
        return None

    @property
    def _target_window_lengths(self):
        # Darts expects an iterable of integers representing lookback/lookahead windows
        return [0]

    @property
    def extreme_lags(self):
        return self.input_chunk_length, 0, 0, 0, 0, 0

    @property
    def min_train_samples(self):
        return self.input_chunk_length + self.output_chunk_length

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_transferable_series_prediction(self) -> bool:
        return False

    def fit(self, series: TimeSeries, past_covariates: TimeSeries = None, val_series: TimeSeries = None,
            val_past_covariates: TimeSeries = None):
        #super().fit(series)
        self.training_series = series

        val_size = 0
        if val_series is not None:
            val_size = len(val_series)
            # Combine training and validation for Nixtla's val_size logic
            full_series = series.append(val_series)
            full_past_covariates = None
            if past_covariates is not None or val_past_covariates is not None:
                # Ensure we have covariates for the full duration
                pc = past_covariates if past_covariates is not None else series.with_values(np.zeros((len(series), 0)))
                vpc = val_past_covariates if val_past_covariates is not None else val_series.with_values(
                    np.zeros((len(val_series), 0)))
                full_past_covariates = pc.append(vpc)

            df, hist_exog = self._to_nixtla_format(full_series, full_past_covariates)
        else:
            df, hist_exog = self._to_nixtla_format(series, past_covariates)

        # PyTorch Lightning hyperparameter inspection quirk
        input_chunk_length = self.input_chunk_length
        output_chunk_length = self.output_chunk_length
        n_epochs = self.n_epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        device = self.device
        print(f"Current device: {device}")

        add_encoders = self.add_encoders

        # Filter tcn_kwargs to exclude parameters intended for the model but not the trainer
        tcn_kwargs = self.tcn_kwargs.copy()
        for k, v in tcn_kwargs.items():
            locals()[k] = v

        if device == "cuda":
            tcn_kwargs["accelerator"] = "gpu"
            tcn_kwargs["devices"] = 1
            tcn_kwargs["logger"] = False
        tcn = TCN(
            h=output_chunk_length,
            input_size=input_chunk_length,
            max_steps=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hist_exog_list=hist_exog,
            alias="TCN",
            **tcn_kwargs
        )

        self.model = NeuralForecast(
            models=[tcn],
            freq=series.freq_str
        )

        # Determine target columns for NeuralForecast
        if len(series.columns) == 1:
            self.model.fit(df=df, val_size=val_size, target_col="y")
        else:
            # Don't pass target_col for multivariate - let NeuralForecast auto-detect
            self.model.fit(df=df, val_size=val_size)

        return self

    def predict(self, n: int, series: TimeSeries = None, past_covariates: TimeSeries = None, **kwargs):
        if series is None:
            series = self.training_series

        df, _ = self._to_nixtla_format(series, past_covariates)
        for m in self.model.models:
            m.trainer_kwargs['logger'] = False
        forecast_df = self.model.predict(df=df)

        # Extract forecast values based on whether series is univariate or multivariate
        if len(series.columns) == 1:
            # Univariate: NeuralForecast outputs a column named after the model alias
            forecast_values = forecast_df["TCN"].values[:n]
        else:
            # Multivariate: NeuralForecast appends model alias to each target column
            # Try both original names and alias-suffixed names
            forecast_cols = []
            for col in series.columns:
                if f"{col}-TCN" in forecast_df.columns:
                    forecast_cols.append(f"{col}-TCN")
                elif col in forecast_df.columns:
                    forecast_cols.append(col)
                else:
                    raise ValueError(f"Cannot find forecast column for '{col}' in output")

            forecast_values = forecast_df[forecast_cols].values[:n]

        # Construct the time index starting from after the input series
        forecast_index = pd.date_range(
            start=series.end_time() + series.freq,
            periods=len(forecast_values),
            freq=series.freq
        )

        return TimeSeries.from_times_and_values(
            times=forecast_index,
            values=forecast_values,
            columns=series.columns
        )

    def historical_forecasts(
            self,
            series: TimeSeries,
            past_covariates: TimeSeries = None,
            start: pd.Timestamp = None,
            forecast_horizon: int = 1,
            stride: int = 1,
            last_points_only: bool = False,
            retrain: bool = False,
            verbose: bool = False,
            **kwargs
    ):
        df, _ = self._to_nixtla_format(series, past_covariates)

        for m in self.model.models:
            m.trainer_kwargs['logger'] = False

        # Nixtla's cross_validation anchors from the END of the series backwards.
        # We calculate n_windows so that the earliest cutoff aligns with `start`.
        total_len = len(series)
        start_idx = series.time_index.get_loc(start)
        # Number of steps from start to the last valid cutoff (total_len - forecast_horizon)
        n_windows = (total_len - start_idx - forecast_horizon) // stride + 1

        cv_df = self.model.cross_validation(
            df=df,
            h=forecast_horizon,
            step_size=stride,
            n_windows=n_windows
        )

        # Convert the Nixtla CV output back to a list of Darts TimeSeries
        forecasts = []
        unique_cutoffs = sorted(cv_df['cutoff'].unique())

        for cutoff in unique_cutoffs:
            window_df = cv_df[cv_df['cutoff'] == cutoff].copy()

            # Extract forecast values based on whether series is univariate or multivariate
            if len(series.columns) == 1:
                # Univariate: extract the model alias column
                values = window_df["TCN"].values.reshape(-1, 1)
                columns = series.columns
            else:
                # Multivariate: extract all target columns with their alias suffixes
                forecast_cols = []
                for col in series.columns:
                    if f"{col}-TCN" in window_df.columns:
                        forecast_cols.append(f"{col}-TCN")
                    elif col in window_df.columns:
                        forecast_cols.append(col)
                    else:
                        raise ValueError(f"Cannot find forecast column for '{col}' in cross-validation output")

                values = window_df[forecast_cols].values
                columns = series.columns

            times = pd.date_range(
                start=pd.Timestamp(cutoff) + series.freq,
                periods=len(window_df),
                freq=series.freq
            )

            ts = TimeSeries.from_times_and_values(
                times=times,
                values=values,
                columns=columns
            )
            forecasts.append(ts)

        if last_points_only:
            # Return one TimeSeries made of the last predicted point of each window
            last_points = [ts[-1] for ts in forecasts]
            if len(last_points) == 1:
                return last_points[0]
            return last_points[0].concatenate(last_points[1:], axis=0)

        return forecasts


    def save(self, path: str, **kwargs):
        if self.model is not None:
            self.model.save(path)
        else:
            raise ValueError("Model has not been fitted yet.")

    @classmethod
    def load(cls, path: str):
        nf_model = NeuralForecast.load(path)


        tcn_instance = nf_model.models[0]

        # Instantiate the adapter with dummy values initially
        instance = cls(
            input_chunk_length=tcn_instance.input_size,
            output_chunk_length=tcn_instance.h
        )
        instance.model = nf_model
        return instance


    @staticmethod
    def _to_nixtla_format(series: TimeSeries, past_covariates: TimeSeries = None):
        df = series.to_dataframe().reset_index()
        df = df.rename(columns={df.columns[0]: "ds"})

        if len(series.columns) == 1:
            # Univariate: rename to 'y' for Nixtla's default behavior
            df = df.rename(columns={series.columns[0]: "y"})
        else:
            # Multivariate: keep original column names
            # NeuralForecast will treat all non-'ds', non-'unique_id' columns as targets
            # (except those in hist_exog_list if specified)
            pass

        hist_exog_list = None
        if past_covariates is not None:
            cov_df = past_covariates.to_dataframe().reset_index()
            cov_df = cov_df.rename(columns={cov_df.columns[0]: "ds"})
            df = pd.merge(df, cov_df, on="ds", how="left")
            hist_exog_list = list(past_covariates.columns)

        df["unique_id"] = "series_1"
        return df, hist_exog_list


def extract_horizon(configuration):
    return configuration[PARAM]["output_chunk_length"]


def extract_lags(configuration):
    return configuration[PARAM]["input_chunk_length"]


def get_tcn_configuration(exp, st_lt):
    configuration_tcn = {
        TYPE: Models.TCN,
        CONSTRUCTOR: NixtlaTCN,
        LAGS: extract_lags,
        PREDICTION_HORIZON: extract_horizon,
        P_GET: get_tcn_param
    }
    return configuration_tcn


def get_tcn_param(input_sizes, horizons, past=True):
    # https://github.com/decisionintelligence/TFB/blob/master/ts_benchmark/baselines/moderntcn/moderntcn.py

    return  {
        "input_chunk_length": input_sizes,
        "output_chunk_length": horizons,
        "n_epochs": 100,               # num_epochs
        "batch_size": 128,             # batch_size
        "learning_rate": 0.0001,       # lr
        "kernel_size": 25,             # kernel_size
        "dilations": [1, 2, 4, 8, 16], # Standard expansion
        "encoder_hidden_size": 256,    # dims[0]
        "decoder_hidden_size": 256,    # Matching encoder capacity
        "decoder_layers": 2,           # Standard depth
    }