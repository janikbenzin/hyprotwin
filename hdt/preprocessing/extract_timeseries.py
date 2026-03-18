import pm4py
import traceback

from hdt.util import *
import pandas as pd


def resample_impute_save(df):
    df[TIME] = pd.to_datetime(df[TIME])
    df = df.set_index(TIME)
    df = df.sort_index()


    df_lin = df.resample(SAMPLING_RATE).ffill()
    if PREPROCESSING_INST:
        df_lin = df_lin.ffill()
        df_lin = df_lin.dropna()
    resampled = pd.DataFrame({"timestamp": df_lin.index})
    orig = pd.DataFrame({"orig_ts": df.index})
    merged = pd.merge_asof(resampled, orig, left_on="timestamp", right_on="orig_ts",
                           direction="backward")
    merged = merged.set_index("timestamp")
    df_lin[DIFFERENCE_TS] = (df_lin.index.to_series() - merged["orig_ts"]).dt.total_seconds()

    if RUN_SCENARIO:
        df_lin.to_csv(get_constant_dataset(exp, "scenario/"))
    else:
        df_lin.to_csv(get_constant_dataset(exp))


def get_diff_cells(df1, df2):
    common_index = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)

    df1_subset = df1.loc[common_index, common_columns]
    df2_subset = df2.loc[common_index, common_columns]

    diff_mask = ~((df1_subset == df2_subset) | (df1_subset.isna() & df2_subset.isna()))

    diffs = []
    for col in common_columns:
        col_diffs = diff_mask[col]
        if col_diffs.any():
            diff_indices = col_diffs[col_diffs].index
            for idx in diff_indices:
                diffs.append({
                    'row_index': idx,
                    'column': col,
                    'df1_value': df1_subset.loc[idx, col],
                    'df2_value': df2_subset.loc[idx, col]
                })

    return pd.DataFrame(diffs)

def get_first_decision(log):
    first_ts_mode_decision = log[((log["cpee:activity"] == "external") & (
            log["cpee:lifecycle:transition"] == "gateway/decide"))]["time:timestamp"].to_list()[0]
    times = list(df["time:timestamp"])
    # For sensor values with timestamps before the first decision in the control process, a dedicated initalization of the corresponding control modes must be done
    before = [ts for ts in times if ts < first_ts_mode_decision]
    return times, before


def fill_initial_nans_with_default(combined_df2, mode_column_name, exp):
    first_non_nan_idx = combined_df2[mode_column_name].first_valid_index()

    if first_non_nan_idx is None:
        combined_df2[mode_column_name] = INITIAL_MODE_VALUE[exp]
    else:
        combined_df2.loc[:first_non_nan_idx, mode_column_name] = \
            combined_df2.loc[:first_non_nan_idx, mode_column_name].fillna(INITIAL_MODE_VALUE[exp])

    return combined_df2


if __name__ == "__main__":
    for exp in [HEATING,WATERLEVEL,IRRIGATION]:
        if exp != SWAT:
            dir_path = get_data_path(exp)
            if RUN_SCENARIO:
                dir_path = dir_path + "scenario/"
                subprocesses, t = extract_sub_from_index(dir_path)
            else:
                subprocesses, t = extract_sub_from_index(dir_path)
            for i, (sid, sname) in enumerate(subprocesses.items()):
                filename = dir_path + get_log_name(sid)
                if "1x" not in sname and "Control" in sname:
                    log = pm4py.read_yaml(filename)
                    if exp.startswith("heating") and sname.startswith("Heating - Control"):
                        df = log[((log["concept:name"] == "Read") & (
                                    log["cpee:lifecycle:transition"] == "dataelements/change"))]
                        ## The following modes identification includes the initialization in which no cycle of the control
                        # For initialization, we require the first decision ts
                        times, before = get_first_decision(log)
                        # before the first control condition evaluation through a decision, the current mode is undetermined for the control process
                        # hence we prune the corresponding temperature values from the time-series, because no pertaining control mode can be extracted
                        if PREPROCESSING_INST:
                            mode_df = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                                    log["cpee:lifecycle:transition"] == "activity/done")]
                            combined_df = pd.merge(mode_df, df, on='time:timestamp', how='outer')
                            combined_df["data_y"] = combined_df["data_y"].apply(
                                lambda cell: cell["children"][0][1]["children"]["value"] if not isinstance(cell,
                                                                                                           float) else np.nan)
                            combined_df2 = combined_df[["time:timestamp", "concept:name_x", "data_y"]]
                            if RUN_SCENARIO:
                                combined_df2 = fill_initial_nans_with_default(combined_df2, "concept:name_x", exp)
                            else:
                                combined_df2 = combined_df2.iloc[len(before):]
                            df = combined_df2.rename(
                                columns={"time:timestamp": TIME, "concept:name_x": get_mode_name(exp, 0),
                                         "data_y": SENSOR_PREFIX + TEMP})
                        else:
                            pruned_times = times[len(before):]
                            temperatures = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))[len(before):]
                            setting_modes = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                                    log["cpee:lifecycle:transition"] == "activity/done")].to_dict('list')
                            if setting_modes["time:timestamp"][0] > pruned_times[0]:
                                # special case in which the first finished control mode activity is after the first timestamp
                                print("Implement special routine")
                            modes = []
                            for i, tsc in enumerate(pruned_times):
                                for j, ts in enumerate(setting_modes["time:timestamp"]):
                                    if j + 1 == len(setting_modes["time:timestamp"]):
                                        modes.append(setting_modes["concept:name"][j])
                                        break
                                    if ts <= tsc < setting_modes["time:timestamp"][j + 1]:
                                        modes.append(setting_modes["concept:name"][j])
                                        break


                            variables_all[exp] = {TIME: [str(j) for j in pruned_times], SENSOR_PREFIX + TEMP: temperatures, get_mode_name(exp, 0): modes}
                            df = pd.DataFrame(data={SENSOR_PREFIX + TEMP: variables_all[exp][SENSOR_PREFIX + TEMP],
                                                    get_mode_name(exp, 0): variables_all[exp][get_mode_name(exp, 0)],
                                                    TIME: variables_all[exp][TIME]})

                        resample_impute_save(df)
                    elif exp.startswith("waterlevel") and sname.startswith("Waterlevel - Control"):
                        df = log[((log["concept:name"] == "Read") & (
                                        log["cpee:lifecycle:transition"] == "dataelements/change"))]
                        mode_df = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                                log["cpee:lifecycle:transition"] == "activity/done")]
                        #times = [str(j) for j in list(df["time:timestamp"])]
                        #levels = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))
                        #variables_all[exp][sid] = {TIME: times, SENSOR_PREFIX + TEMP: levels}
                        #mode_df = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                        #        log["cpee:lifecycle:transition"] == "activity/done")]
                        combined_df = pd.merge(mode_df, df, on='time:timestamp', how='outer')
                        combined_df["data_y"] = combined_df["data_y"].apply(
                            lambda cell: cell["children"][0][1]["children"]["value"] if not isinstance(cell,
                                                                                                       float) else np.nan)
                        combined_df2 = combined_df[["time:timestamp", "concept:name_x", "data_y"]]
                        times, before = get_first_decision(log)
                        if RUN_SCENARIO:
                            combined_df2 = fill_initial_nans_with_default(combined_df2, "concept:name_x", exp)
                        else:
                            combined_df2 = combined_df2.iloc[len(before):]
                        df = combined_df2.rename(
                            columns={"time:timestamp": TIME, "concept:name_x": get_mode_name(exp, 0),
                                     "data_y": SENSOR_PREFIX + LEVEL})
                        resample_impute_save(df)
                    elif exp.startswith("irrigation") and sname.startswith("Soil Moisture - Control"):
                        df = log[((log["concept:name"] == "Read") & (
                                log["cpee:lifecycle:transition"] == "dataelements/change"))]
                        mode_df = log[((log["concept:name"] == "Irrigation On") | (log["concept:name"] == "Irrigation Off")) & (
                                log["cpee:lifecycle:transition"] == "activity/done")]
                        # times = [str(j) for j in list(df["time:timestamp"])]
                        # levels = list(df["data"].apply(lambda cell: cell["children"][0][1]["children"]["value"]))
                        # variables_all[exp][sid] = {TIME: times, SENSOR_PREFIX + TEMP: levels}
                        # mode_df = log[((log["concept:name"] == "On") | (log["concept:name"] == "Off")) & (
                        #        log["cpee:lifecycle:transition"] == "activity/done")]
                        combined_df = pd.merge(mode_df, df, on='time:timestamp', how='outer')
                        combined_df["data_y"] = combined_df["data_y"].apply(
                            lambda cell: cell["children"][0][1]["children"]["value"] if not isinstance(cell,
                                                                                                       float) else np.nan)
                        combined_df2 = combined_df[["time:timestamp", "concept:name_x", "data_y"]]
                        times, before = get_first_decision(log)
                        if RUN_SCENARIO:
                            combined_df2 = fill_initial_nans_with_default(combined_df2, "concept:name_x", exp)
                        else:
                            combined_df2 = combined_df2.iloc[len(before):]
                        df = combined_df2.rename(
                            columns={"time:timestamp": TIME, "concept:name_x": get_mode_name(exp, 0),
                                     "data_y": SENSOR_PREFIX + MOIST})
                        resample_impute_save(df)

                                #variables_all[exp][sid] = {TIME: times, SENSOR_PREFIX + TEMP: levels}
        else:
            df = pd.read_csv("./data/swat_a1a2/Physical/swat_normal.csv", low_memory=False, header=0)

            df.columns = df.iloc[0].str.strip()

            #  Process the Timestamp: Convert to datetime and set as index
            # The column name is "Timestamp" based on your selection
            df = df.iloc[1:].reset_index(drop=True)
            df = df.rename(columns={'Timestamp': TIME})
            df[TIME] = pd.to_datetime(df[TIME].str.strip())

            df.set_index(TIME, inplace=True)

            df = df.drop(columns=['P102', 'P202', 'P204', 'P206', 'P301', 'P401', 'P404', 'P502', 'P601', 'P603'])
            #  Define the two lists of names
            # Sensors: Flow (FIT), Level (LIT), Analysers (AIT), Pressure (PIT/DPIT)
            sensor_columns = [col for col in df.columns if
                              any(suffix in col for suffix in ['FIT', 'LIT', 'AIT', 'PIT', 'DPIT'])]

            # Actuators: Motorized Valves (MV), Pumps (P), UV Dechlorinator (UV)
            actuator_columns = [col for col in df.columns if any(prefix in col and 'DPIT' not in col and 'PIT' not in col for prefix in ['MV', 'P', 'UV'])]

            # Divide into two separate dataframes and cast to float
            #df_sensors = df[sensor_columns].astype(float)
            #df_actuators = df[actuator_columns].astype(float)

            # Display results
            print(f"Sensor Columns ({len(sensor_columns)}):", sensor_columns)
            print(f"Actuator Columns ({len(actuator_columns)}):", actuator_columns)

            #variables_all[exp] = {TIME: df.index.to_list(), SENSOR_PREFIX + TEMP: temperatures,
            #                      get_mode_name(exp, 0): modes}
            df = df.rename(columns={old: f"{SENSOR_PREFIX}{old}" for old in sensor_columns})
            df = df.rename(columns={old: f"{MODE_PREFIX}{old}" for old in actuator_columns})
            df.to_csv(get_constant_dataset(exp))

    #with open(get_raw_datasets(), "w") as f:
    #    json.dump(variables_all, f)

