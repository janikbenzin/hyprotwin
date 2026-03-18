from typing import Any

from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
from sklearn.preprocessing import StandardScaler
from darts.utils.model_selection import train_test_split
import numpy as np
from pandas import Timedelta
from pandas._libs import NaTType

from hdt.util import *
from hdt.preprocessing.extract_timeseries import TIME, SENSOR_PREFIX, MODE_PREFIX, DIFFERENCE_TS
from hdt.run.configuration import SeriesBundle
from hdt.util import _load_pickle


def preprocess_exp(exp, fold):
    if exp in [HEATING, WATERLEVEL, IRRIGATION]:
        next_cutoff, scale_max, scale_min, series_scaled, test_scaled, train_scaled, val_scaled = apply(exp, fold)
    elif exp == SWAT:
        next_cutoff, scale_max, scale_min, series_scaled, test_scaled, train_scaled, val_scaled = apply(exp, fold, scale_t="zscore")
    else:
        next_cutoff, scale_max, scale_min, series_scaled, test_scaled, train_scaled, val_scaled = tuple([None] * 7)

    print(f"Finished preprocessing with CV fold {fold} and cutoff time {next_cutoff} resulting in Train size: {len(train_scaled)} Val size: {len(val_scaled)} Test size: {len(test_scaled)}")
    return series_scaled, train_scaled, val_scaled, test_scaled, scale_min, scale_max


def apply(exp, fold, scale_t="minmax") -> tuple[
    Timedelta | NaTType | Any, TimeSeries | list[TimeSeries], TimeSeries | list[TimeSeries], TimeSeries | list[
        TimeSeries], TimeSeries | list[TimeSeries], list[Any], list[Any]]:
    df_lin = pd.read_csv(get_constant_dataset(exp), index_col=TIME, parse_dates=True)
    df_lin = df_lin.dropna()
    df_lin = df_lin.map(lambda cell: encode_modes(exp, cell) if not isinstance(cell, float) else cell)
    filler = MissingValuesFiller()
    if scale_t == "minmax":
        scaler = Scaler()
    else:
        s = StandardScaler()
        scaler = Scaler(s)
    series = filler.transform(
        TimeSeries.from_dataframe(
            df_lin,
            value_cols=[var for var in df_lin.columns.to_list() if var.startswith(SENSOR_PREFIX)] + get_all_mode_names(
                exp),
            freq=SAMPLING_RATE
        )
    ).astype(np.float32)
    train, test = train_test_split(series, input_size=1, horizon=1, test_size=0.2, lazy=True)
    cutoff = test.start_time()
    train_tmp, val = train_test_split(train, input_size=1, horizon=1, test_size=.125, lazy=True)
    val_size = len(val)
    # USED FOR TIME-SERIES CV
    next_cutoff = cutoff + pd.Timedelta(1800 * fold, "s")
    test = series.drop_before(next_cutoff, keep_point=True)
    val = series.drop_after(next_cutoff)[-val_size:]
    train = series.drop_after(next_cutoff)[:-val_size]
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    series_scaled = scaler.transform(series)
    if scale_t == "minmax":
        scale_min = list(train[[c for c in df_lin.columns if c.startswith(SENSOR_PREFIX)]].min(axis=0).values()[0])
        scale_max = list(train[[c for c in df_lin.columns if c.startswith(SENSOR_PREFIX)]].max(axis=0).values()[0])
    else:
        fitted_scaler = scaler._fitted_params[0]
        scale_min = fitted_scaler.mean_
        scale_max = fitted_scaler.scale_
    return next_cutoff, scale_max, scale_min, series_scaled, test_scaled, train_scaled, val_scaled


def load_or_prepare_scenario_bundle(forecast_config, eval_cfg, data_bundle, fold):
    bundle_file_scenario = get_bundles_path(forecast_config.exp, "scenario")
    if not os.path.exists(bundle_file_scenario):
        bundle_file = get_bundles_path(forecast_config.exp, fold)
        if not os.path.exists(bundle_file):
            series_scaled, train_scaled, val_scaled, test_scaled, scale_min, scale_max = preprocess_exp(forecast_config.exp, fold)
            data_bundle = SeriesBundle(train_scaled, val_scaled, test_scaled)
            with open(bundle_file, "wb") as f:
                pickle.dump((data_bundle, scale_min, scale_max), f)
        else:
            with open(bundle_file, "rb") as f:
                data_bundle, scale_min, scale_max = pickle.load(f)
                data_bundle = SeriesBundle(
                    data_bundle.train_scaled.astype(np.float32),
                    data_bundle.val_scaled.astype(np.float32),
                    data_bundle.test_scaled.astype(np.float32)
                )
        if DIFFERENCE_TS in data_bundle.train_scaled.components:
            data_bundle = SeriesBundle(train_scaled=data_bundle.train_scaled.drop_columns(DIFFERENCE_TS),
                                       val_scaled=data_bundle.val_scaled.drop_columns(DIFFERENCE_TS),
                                       test_scaled=data_bundle.test_scaled.drop_columns(DIFFERENCE_TS))
        test_scenario_scaled = apply_scenario(forecast_config.exp, forecast_config, eval_cfg)
        new_test_scaled = replace_last_values(data_bundle.test_scaled, test_scenario_scaled, SCENARIO_TIMES[forecast_config.exp])
        data_bundle = replace(data_bundle, test_scaled=new_test_scaled)
        with open(bundle_file_scenario, "wb") as f:
            pickle.dump((data_bundle, eval_cfg.scale_min, eval_cfg.scale_max), f)
    else:
        data_bundle, scale_min, scale_max = _load_pickle(bundle_file_scenario)
        data_bundle = replace(data_bundle, train_scaled=data_bundle.train_scaled.astype(np.float32),
                              val_scaled=data_bundle.val_scaled.astype(np.float32),
                              test_scaled=data_bundle.test_scaled.astype(np.float32))
        #data_bundle = SeriesBundle(
        #    data_bundle.train_scaled.astype(np.float32),
         ##
         #
        #)
    return data_bundle


def apply_scenario(exp, fc_cfg, eval_cfg):
    df_lin = pd.read_csv(get_constant_dataset(exp, "scenario/"), index_col=TIME, parse_dates=True)
    df_lin = df_lin.dropna()
    df_lin = df_lin.map(lambda cell: encode_modes(exp, cell) if not isinstance(cell, float) else cell)
    filler = MissingValuesFiller()
    scale_min = eval_cfg.scale_min[0]
    scale_max = eval_cfg.scale_max[0]
    series = filler.transform(
        TimeSeries.from_dataframe(
            df_lin,
            value_cols=[var for var in df_lin.columns.to_list() if
                        var.startswith(SENSOR_PREFIX)] + get_all_mode_names(
                exp),
            freq=SAMPLING_RATE
        )
    ).astype(np.float32)
    series_sensors = get_sensors_only(series, fc_cfg)
    series_sensors = minmax_scale_series(series_sensors, scale_min, scale_max)
    series_modes = series[get_all_mode_names(exp)]
    series = concatenate([series_sensors, series_modes], axis=1)
    return series


def minmax_scale_series(ts: TimeSeries, scale_min: np.ndarray, scale_max: np.ndarray) -> TimeSeries:
    vals = ts.values(copy=True)  # shape: (time, components, samples)
    denom = (scale_max - scale_min)
    denom = np.where(denom == 0, 1.0, denom)
    vals = (vals - scale_min) / denom
    return ts.with_values(vals)

def minmax_inverse_series(ts_scaled: TimeSeries, scale_min: np.ndarray, scale_max: np.ndarray) -> TimeSeries:
    vals = ts_scaled.values(copy=True)
    vals = vals * (scale_max - scale_min) + scale_min
    return ts_scaled.with_values(vals)

def build_segmented_dataset(target_segments,
                            covariate_segments,
                            lags_target,
                            lags_cov=None):

    if lags_cov is None:
        lags_cov = lags_target

    X_list = []
    Y_list = []

    for seg_target, seg_cov in zip(target_segments, covariate_segments):
        tgt = seg_target.values()
        cov = seg_cov.values()

        T = tgt.shape[0]
        max_lag = max(max(lags_target), max(lags_cov))

        for t in range(max_lag, T):
            features = []

            for lag in lags_target:
                features.extend(tgt[t - lag])

            for lag in lags_cov:
                features.extend(cov[t - lag])

            Y_list.append(tgt[t])

            X_list.append(features)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(Y_list, dtype=np.float32)

    return X, y
