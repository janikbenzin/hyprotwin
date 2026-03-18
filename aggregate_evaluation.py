import json
import os
import re
import datetime
import pandas as pd
import numpy as np

# Import constants from parameters
from hdt.parameters import (
    HEATING,
    WATERLEVEL,
    IRRIGATION,
    RUN_PREDICTION_STRIDES,
    RUN_PREDICTION_HORIZONS,
    RUN_INPUT_SIZES,
    Models,
    Scores,
    Horizon,
    Segmentation,
    Components,
    PATH_PREFIX
)

RESULTS_DIR = os.path.join(PATH_PREFIX, "tmp")

EXPERIMENTS = [HEATING, WATERLEVEL, IRRIGATION]

# Extract model values from enum
MODEL_VALUES = [m.value for m in Models]

# Extract score keys from enum
SCORE_KEYS = [s.value for s in Scores]

# Each variant describes how to navigate the all_intermediate tree.
# "horizon"   : Horizon enum value
# "seg"       : Segmentation enum value
# "component" : Components enum value
# "mode_key"  : if key is None, then we have non-sim variants, for key = '0.0' we have sim variants
# "scenario"  : whether this represents a future what if scenario


_DATASET_NAME = "Cont. Proc."
_HORIZON = "$T_F$"
_STRIDE = "$I$"
_SIZE = "$T_H$"

#  Variant definitions
VARIANTS = [
    # Long-term NON-SCENARIO (single variant, no sim/non-sim split)
    {
        "label": "Direct",
        "horizon": Horizon.LT,
        "seg": Segmentation.OFF,
        "component": Components.ALL,
        "mode_key": Horizon.LT.value,  # "long-term"
        "scenario": False,
    },
    # Short-term NON-SCENARIO variants
    {
        "label": "HyProTwin",
        "horizon": Horizon.ST,
        "seg": Segmentation.OFF,
        "component": Components.ALL,
        "mode_key": Horizon.ST.value,  # "short-term"
        "scenario": False,
    },
    {
        "label": "HyProTwin (opt)",
        "horizon": Horizon.ST,
        "seg": Segmentation.ON,
        "component": Components.ALL,
        "mode_key": "0.0",
        "scenario": False,
    },
    # Long-term SCENARIO (single variant, no sim/non-sim split)
    {
        "label": "Direct Scenario",
        "horizon": Horizon.LT,
        "seg": Segmentation.OFF,
        "component": Components.SENSORS_ONLY,  # Scenario uses SENSORS_ONLY
        "mode_key": Horizon.LT.value,
        "scenario": True,
    },
    #Short-term SCENARIO variants
    {
        "label": "HyProTwin Scenario",
        "horizon": Horizon.ST,
        "seg": Segmentation.OFF,
        "component": Components.SENSORS_ONLY,
        "mode_key": Horizon.ST.value,
        "scenario": True,
    },
    {
        "label": "HyProTwin Scenario (opt)",
        "horizon": Horizon.ST,
        "seg": Segmentation.ON,
        "component": Components.SENSORS_ONLY,
        "mode_key": "0.0",
        "scenario": True,
    },
]


def load_latest_final_results(path_prefix: str = RESULTS_DIR, specific=None) -> dict:
    pattern = re.compile(r"final_results_dict_(.+)\.json")
    latest_file, latest_ts = None, None

    for fname in os.listdir(path_prefix):
        m = pattern.match(fname)
        if not m:
            continue
        try:
            ts = datetime.datetime.fromisoformat(m.group(1))
        except ValueError:
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts, latest_file = ts, os.path.join(path_prefix, fname)

    if latest_file is None:
        raise FileNotFoundError(f"No final_results_dict_*.json found in {path_prefix}")

    if specific is not None:
        latest_file = path_prefix + "/" + specific
    with open(latest_file) as f:
        data = json.load(f)
    print(f"Loaded: {latest_file}")
    return data


def _safe_avg(value):
    if value is None:
        return float("nan")
    if isinstance(value, list):
        valid = [v for v in value if v is not None]
        return float(np.mean(valid)) if valid else float("nan")
    return float(value)


def extract_scores(all_intermediate: dict) -> pd.DataFrame:
    rows = []

    for exp in EXPERIMENTS:
        if exp not in all_intermediate:
            continue
        exp_data = all_intermediate[exp]

        # Smallest stride for this experiment (only this one is recorded)
        min_stride = str(min(RUN_PREDICTION_STRIDES[exp]))

        for model_val in MODEL_VALUES:
            if model_val not in exp_data:
                continue
            model_data = exp_data[model_val]

            for variant in VARIANTS:
                seg_val = variant["seg"].value
                comp_val = variant["component"].value
                mode_key = variant["mode_key"]
                label = variant["label"]
                scenario = variant["scenario"]

                try:
                    node = model_data[seg_val][comp_val][mode_key]
                except KeyError:
                    continue

                for horizon in RUN_PREDICTION_HORIZONS[exp]:
                    h_str = str(horizon)
                    if h_str not in node:
                        continue

                    for input_size in RUN_INPUT_SIZES[exp][horizon]:
                        is_str = str(input_size)
                        if is_str not in node[h_str]:
                            continue

                        stride_node = node[h_str][is_str].get(min_stride)
                        if stride_node is None:
                            continue

                        score_node = stride_node.get("scores", {})
                        row = {
                            _DATASET_NAME: exp,
                            "f": model_val,
                            "variant": label,
                            "scenario": "Yes" if scenario else "No",
                            _HORIZON: horizon,
                            _SIZE: input_size,
                            _STRIDE: int(min_stride),
                        }
                        for sk in SCORE_KEYS:
                            row[sk] = _safe_avg(score_node.get(sk))

                        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        _DATASET_NAME, "f", "variant", "scenario",
        _HORIZON, _SIZE, _STRIDE,
        *SCORE_KEYS,
    ])
    # df["f"] = df["f"].apply(lambda cell: FINAL_MODEL_NAMES[cell])
    return df


# Nicer display names
_EXP_NAMES = {HEATING: "Heating", WATERLEVEL: "Water tank", IRRIGATION: "Irrigation"}
_MODEL_NAMES = {
    Models.LightGBM.value: "GBM",
    Models.DLinear.value: "DLin",
    Models.LSTM.value: "LSTM",
    Models.TCN.value: "TCN"
}



def build_latex_table(df: pd.DataFrame, score: str = "RMSE") -> str:
    if df.empty:
        return "% DataFrame is empty – no scores to display."

    pivot = df.pivot_table(
        index=[_DATASET_NAME, "f", "scenario", "variant"],
        columns=[_HORIZON, _SIZE],
        values=score,
        aggfunc="mean",
    )

    # Rename index levels for readability
    pivot.index = pivot.index.set_levels(
        [pivot.index.levels[0].map(lambda x: _EXP_NAMES.get(x, x)),
         pivot.index.levels[1].map(lambda x: _MODEL_NAMES.get(x, x)),
         pivot.index.levels[2],  # Keep "Yes"/"No" as-is
         pivot.index.levels[3].map(lambda x: x)],
    )
    pivot.index.names = ["Dataset", "Model", "Scenario", "Variant"]

    # Flatten column multi-index: "H=300 / IS=300"
    pivot.columns = [
        f"H={h}\\,/\\,IS={s}" for h, s in pivot.columns
    ]
    pivot.columns.name = None

    latex = pivot.to_latex(
        float_format="%.4f",
        na_rep="--",
        caption=(
            f"Results"
        ),
        label=f"tab:results_{score.lower()}",
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False,
    )
    return latex


def build_comparison_df(df: pd.DataFrame, score: str = "RMSE", scenario=True) -> pd.DataFrame:
    if scenario:
        variants_of_interest = ["Direct Scenario",  "HyProTwin Scenario",  "HyProTwin Scenario (opt)"]
    else:
        variants_of_interest = ["Direct", "HyProTwin", "HyProTwin (opt)"]
    key_cols = [_DATASET_NAME, "f", _HORIZON, _SIZE]

    sub = df[df["variant"].isin(variants_of_interest)]

    pivot = (
        sub.pivot_table(
            index=key_cols,
            columns="variant",
            values=score,
            aggfunc="mean",
        )
        .reindex(columns=variants_of_interest)
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def build_custom_layout_latex(df: pd.DataFrame) -> str:

    scenario_variants = ["Direct Scenario", "HyProTwin Scenario", "HyProTwin Scenario (opt)"]
    sub = df[df["variant"].isin(scenario_variants)].copy()


    sub[_DATASET_NAME] = sub[_DATASET_NAME].map(lambda x: _EXP_NAMES.get(x, x))
    sub["f"] = sub["f"].map(lambda x: _MODEL_NAMES.get(x, x))

    # Define custom orders
    dataset_order = ["Water tank", "Heating", "Irrigation"]
    model_order = ["GBM", "DLin", "LSTM", "TCN"]
    variant_order = ["Direct Scenario", "HyProTwin Scenario", "HyProTwin Scenario (opt)"]
    metric_order = ["RMSE", "MAE"]

    sub[_DATASET_NAME] = pd.Categorical(sub[_DATASET_NAME], categories=dataset_order, ordered=True)
    sub["f"] = pd.Categorical(sub["f"], categories=model_order, ordered=True)

    sub = sub.sort_values([_DATASET_NAME, _HORIZON, _SIZE])

    sub["$T_H/T_F$"] = sub.apply(lambda r: f"{r[_SIZE]}/{r[_HORIZON]}", axis=1)
    th_tf_order = sub["$T_H/T_F$"].unique().tolist()
    sub["$T_H/T_F$"] = pd.Categorical(sub["$T_H/T_F$"], categories=th_tf_order, ordered=True)

    melted = sub.melt(
        id_vars=[_DATASET_NAME, "f", "$T_H/T_F$", "variant"],
        value_vars=["RMSE", "MAE"],
        var_name="Metrics",
        value_name="Score"
    )

    melted["Metrics"] = pd.Categorical(melted["Metrics"], categories=metric_order, ordered=True)

    pivot = melted.pivot_table(
        index=[_DATASET_NAME, "$T_H/T_F$", "Metrics"],
        columns=["f", "variant"],
        values="Score",
        aggfunc="mean",
        sort=False
    )


    target_cols = pd.MultiIndex.from_product([model_order, variant_order], names=["f", "variant"])


    pivot = pivot.reindex(columns=target_cols)

    formatted_pivot = pivot.applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")

    for idx in pivot.index:
        row_values = pivot.loc[idx]
        if row_values.isnull().all():
            continue

        row_min = row_values.min()

        for model in model_order:
            model_slice = row_values.loc[model]
            if model_slice.isnull().all():
                continue

            model_min = model_slice.min()

            for variant in variant_order:
                val = pivot.loc[idx, (model, variant)]
                if pd.isnull(val):
                    continue

                cell_str = f"{val:.4f}"

                # Apply overall best (Bold)
                if val == row_min:
                    cell_str = f"$\\mathbf{{{cell_str}}}$"
                # Apply model-specific best (Underline)
                elif val == model_min:
                    cell_str = f"\\underline{{{cell_str}}}"

                formatted_pivot.loc[idx, (model, variant)] = cell_str

    variant_map = {
        "Direct Scenario": "Direct Forecasting",
        "HyProTwin Scenario": "HyProTwin",
        "HyProTwin Scenario (opt)": "HyProTwin (opt)"
    }
    formatted_pivot.rename(columns=variant_map, level=1, inplace=True)
    formatted_pivot.index.names = ["Dataset", "$T_H$/$T_F$", "Metrics"]
    formatted_pivot.columns.names = ["Model $f$", None]

    latex = formatted_pivot.to_latex(
        na_rep="--",
        caption="Performance of direct techniques, HyProTwin, and optimized (opt) HyProTwin. Bold values indicate the overall best per row, while underlined values indicate the best within a model group.",
        label="tab:results_predsim",
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
        column_format="lllcccccccccccc"
    )
    return latex


def build_custom_layout_latex_technique(df: pd.DataFrame, second="model",
                                        scenario_variants=["Direct Scenario", "HyProTwin Scenario", "HyProTwin Scenario (opt)"],
                                        caption="Performance of direct forecasting, HyProTwin, and optimized (opt) HyProTwin with respect to forecasting future sensors $\\hat{\\mathbf{Y}}$ in predictive simulation.") -> str:
    sub = df[df["variant"].isin(scenario_variants)].copy()

    sub[_DATASET_NAME] = sub[_DATASET_NAME].map(lambda x: _EXP_NAMES.get(x, x))
    sub["f"] = sub["f"].map(lambda x: _MODEL_NAMES.get(x, x))

    # Define custom orders
    dataset_order = ["Water tank", "Heating", "Irrigation"]
    model_order = ["GBM", "DLin", "LSTM", "TCN"]
    #variant_order = ["Direct Scenario", "HyProTwin Scenario", "HyProTwin Scenario (opt)"]
    variant_order = scenario_variants
    metric_order = ["RMSE", "MAE"]

    sub[_DATASET_NAME] = pd.Categorical(sub[_DATASET_NAME], categories=dataset_order, ordered=True)
    sub["f"] = pd.Categorical(sub["f"], categories=model_order, ordered=True)

    sub = sub.sort_values([_DATASET_NAME, _HORIZON, _SIZE])

    sub["$T_H/T_F$"] = sub.apply(lambda r: f"{r[_SIZE]}/{r[_HORIZON]}", axis=1)
    th_tf_order = sub["$T_H/T_F$"].unique().tolist()
    sub["$T_H/T_F$"] = pd.Categorical(sub["$T_H/T_F$"], categories=th_tf_order, ordered=True)

    melted = sub.melt(
        id_vars=[_DATASET_NAME, "f", "$T_H/T_F$", "variant"],
        value_vars=["RMSE", "MAE"],
        var_name="Metrics",
        value_name="Score"
    )

    melted["Metrics"] = pd.Categorical(melted["Metrics"], categories=metric_order, ordered=True)

    pivot = melted.pivot_table(
        index=[_DATASET_NAME, "$T_H/T_F$", "Metrics"],
        columns=["f", "variant"],
        values="Score",
        aggfunc="mean",
        sort=False
    )

    target_cols = pd.MultiIndex.from_product([model_order, variant_order], names=["f", "variant"])
    pivot = pivot.reindex(columns=target_cols)

    formatted_pivot = pivot.applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")
    if second == "model":
        for idx in pivot.index:
            row_values = pivot.loc[idx]
            if row_values.isnull().all():
                continue

            row_min = row_values.min()

            # Highlight per Model group
            for model in model_order:
                model_slice = row_values.loc[model]
                if model_slice.isnull().all():
                    continue

                model_min = model_slice.min()

                for variant in variant_order:
                    val = pivot.loc[idx, (model, variant)]
                    if pd.isnull(val):
                        continue

                    cell_str = f"{val:.4f}"

                    # Apply overall best (Bold)
                    if val == row_min:
                        cell_str = f"$\\mathbf{{{cell_str}}}$"
                    # Apply model-specific best (Underline)
                    elif val == model_min:
                        cell_str = f"\\underline{{{cell_str}}}"

                    formatted_pivot.loc[idx, (model, variant)] = cell_str
    else:
        target_cols = pd.MultiIndex.from_product([model_order, variant_order], names=["f", "variant"])
        pivot = pivot.reindex(columns=target_cols)

        formatted_pivot = pivot.applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "--")

        for idx in pivot.index:
            row_values = pivot.loc[idx]
            if row_values.isnull().all():
                continue

            # Get unique sorted values to find absolute 1st, 2nd, and 3rd best across the whole row
            sorted_vals = np.sort(row_values.dropna().unique())

            row_min = sorted_vals[0] if len(sorted_vals) > 0 else None
            row_second = sorted_vals[1] if len(sorted_vals) > 1 else None
            row_third = sorted_vals[2] if len(sorted_vals) > 2 else None

            for model in model_order:
                for variant in variant_order:
                    val = pivot.loc[idx, (model, variant)]
                    if pd.isnull(val):
                        continue

                    cell_str = f"{val:.4f}"

                    # Apply absolute best (Bold)
                    if val == row_min:
                        cell_str = f"$\\mathbf{{{cell_str}}}$"
                    # Apply absolute second best (Italic)
                    elif val == row_second:
                        cell_str = f"\\cellcolor{{gray!20}}{{{cell_str}}}"
                    # Apply absolute third best (Light Gray Background)
                    elif val == row_third:
                        cell_str = f"$\\mathit{{{cell_str}}}$"

                    formatted_pivot.loc[idx, (model, variant)] = cell_str
    formatted_pivot.columns = formatted_pivot.columns.swaplevel(0, 1)

    target_cols_swapped = pd.MultiIndex.from_product([variant_order, model_order], names=["variant", "f"])
    formatted_pivot = formatted_pivot.reindex(columns=target_cols_swapped)

    variant_map = {
        "Direct Scenario": "Direct Forecasting",
        "HyProTwin Scenario": "HyProTwin",
        "HyProTwin Scenario (opt)": "HyProTwin (opt)"
    }

    # Rename techniques in the top level
    formatted_pivot.index.names = ["System", "$T_H$/$T_F$", "Metrics"]
    formatted_pivot.rename(index=variant_map, columns=variant_map, level=0, inplace=True)
    formatted_pivot.columns.names = [None, "Model $f$"]

    # Generate LaTeX
    latex = formatted_pivot.to_latex(
        na_rep="--",
        caption=caption,
        label="tab:results_predsim",
        multirow=True,
        multicolumn=True,
        multicolumn_format="c",
        escape=False,
        column_format="lllcccccccccccc"
    )
    return latex

if __name__ == "__main__":
    data = load_latest_final_results()
    if data[HEATING][Models.DLinear.value]['on']['sensors']['0.0']['1800']['1800']['15']['scores']['RMSE'] is None:
        # Fix for the only case in which for optimized models in HyProTwin the non-optimized model has a better validation RMSE than the optimized, so the non-optimized model gets selected for both resulting in similar results
        data[HEATING][Models.DLinear.value]['on']['sensors']['0.0']['1800']['1800']['15']['scores'] = data[HEATING][Models.DLinear.value]['off']['sensors']['short-term']['1800']['1800']['15']['scores']
    if data[WATERLEVEL][Models.TCN.value]['on']['sensors']['0.0']['1800']['1800']['15']['scores']['RMSE'] is None:
        # Fix for the only case in which for optimized models in HyProTwin the non-optimized model has a better validation RMSE than the optimized, so the non-optimized model gets selected for both resulting in similar results
        data[WATERLEVEL][Models.TCN.value]['on']['sensors']['0.0']['1800']['1800']['15']['scores'] = data[WATERLEVEL][Models.TCN.value]['off']['sensors']['short-term']['1800']['1800']['15']['scores']
    df = extract_scores(data)
    df = extract_scores(data)

    for score in SCORE_KEYS:
        comp_df = build_comparison_df(df, score=score)
        comp_df["f"] = comp_df["f"].apply(lambda cell: _MODEL_NAMES[cell])
        #print(comp_df.to_string(index=False))
        #print(build_latex_table(comp_df, score=score))


    comp_rmse = build_comparison_df(df, score="RMSE", scenario=True)
    comp_mae = build_comparison_df(df, score="MAE", scenario=True)

    # Apply model names to both
    for d in [comp_rmse, comp_mae]:
        d["f"] = d["f"].apply(lambda cell: _MODEL_NAMES.get(cell, cell))

    #print(build_custom_layout_latex(df))

    print(build_custom_layout_latex_technique(df, second=""))

