import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as rt


from hdt.preprocessing.preprocess import preprocess_exp, load_or_prepare_scenario_bundle

from hdt.remote_util import *

from hdt.run.configuration import *



model_type = Models.LightGBM
exp = WATERLEVEL
prediction_horizon = 1800
component = Components.ALL
segmentation = Segmentation.OFF
input_size = 1800
st_lt = Horizon.ST
local_onnx_path = get_simple_model_path(
                    exp=exp,
                    mt=model_type,
                    st_lt=st_lt,
                    component=component,
                    segmentation=Segmentation.ON,
                    prediction_horizon=prediction_horizon,
                    input_size=input_size,
                    serialization_t="onnx",
                    mode='0.0'
                )
fc_cfg = ForecastConfig(exp, model_type, st_lt, SENSOR_PREFIX, MODE_PREFIX, DIFFERENCE_TS,
                                         get_all_mode_names(exp))

bundle_file = get_bundles_path(exp, 0)
if not os.path.exists(bundle_file):
    series_scaled, train_scaled, val_scaled, test_scaled, scale_min, scale_max = preprocess_exp(exp, 0)
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

tp_cfg = TrainPredictConfig(component, 0 ,
                                                            bundle_file,
                                                            prediction_horizon,
                                                            15,
                                                            input_size,
                                                            gs=False,
                                                            sim=True,
                                                            train_boundaries=True,
                                                            lower_vals=COARSE_BOUNDARIES.get(exp)["min_vals"],
                                                            upper_vals=COARSE_BOUNDARIES.get(exp)["max_vals"],
                                                            actual_boundaries=ACTUAL_BOUNDARIES[exp]
                                                            )

mode_component = get_all_mode_names(exp)
eval_cfg = EvalConfig(scale_min, scale_max)
bundle_scenario = load_or_prepare_scenario_bundle(fc_cfg, eval_cfg, data_bundle, 0)
chunks_scenario = construct_sensor_chunks(data_bundle, fc_cfg, tp_cfg)
w_bundle_scenario = construct_worker_bundle(bundle_scenario, chunks_scenario, mode_component)


sns.set_theme(style="whitegrid")

# Convert the Darts TimeSeries to pandas
df = data_bundle.test_scaled.to_dataframe()[["sensor_level", "mode_flow"]].copy()
df["sensor_level"] = df["sensor_level"] * (scale_max[0] - scale_min[0]) + scale_min[0]
# First 2 minutes = 120 seconds
df_2min = df.iloc[:120].copy()


# Elapsed time in seconds for x-axis
df_2min["seconds"] = (df_2min.index - df_2min.index[0]).total_seconds()

fig, ax = plt.subplots(figsize=(7, 6))

# Plot with visible recorded points
sns.lineplot(
    data=df_2min,
    x="seconds",
    y="sensor_level",
    ax=ax,
    label="Sensor level",
    color="tab:blue",
    marker="o",
    markersize=4
)

# Shade intervals where mode_flow == 1
active = df_2min["mode_flow"].eq(1)
span_start = None
span_label_added = False

for i in range(len(df_2min)):
    if active.iloc[i] and span_start is None:
        span_start = df_2min["seconds"].iloc[i]
    elif not active.iloc[i] and span_start is not None:
        span_end = df_2min["seconds"].iloc[i]
        ax.axvspan(
            span_start,
            span_end,
            color="lightgray",
            alpha=0.35,
            label="Actuator Off (closed)" if not span_label_added else None
        )
        span_label_added = True
        span_start = None


if span_start is not None:
    ax.axvspan(
        span_start,
        df_2min["seconds"].iloc[-1],
        color="lightgray",
        alpha=0.35,
        label="mode_flow active" if not span_label_added else None
    )

#ax.set_title("Sensor level readings of the water tank")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Water (litre)")
ax.margins(x=0)
ax.set_xlim(0, df_2min["seconds"].iloc[-1])
ax.legend()

plt.tight_layout()
fig.savefig("waterlevel.svg")
plt.show()