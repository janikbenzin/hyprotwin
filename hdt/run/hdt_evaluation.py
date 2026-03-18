import os.path
from typing import Any

from hdt.preprocessing.preprocess import preprocess_exp, apply_scenario, load_or_prepare_scenario_bundle
from hdt.remote_util import *
import subprocess
import json
import pm4py
import ast
import pandas as pd


import time
from urllib.parse import quote

import requests
from lxml import html as lxml_html

from hdt.run.configuration import SeriesBundle
from hdt.util import select_best_model_by_rmse, custom_json_serializer, _load_pickle, _save_pickle
from hdt.preprocessing.extended_pm4py_yaml_read import read_yaml


def _build_monitor_url(instance_url: str) -> str:
    base = f"https://{target_host}/flow/"
    # Encode the full instance URL into the monitor query parameter
    return f"{base}?monitor={instance_url}/"


def _store_superprocess_log_url(exp: str, model_type, log_url: str, allow_simulation: bool = False):
    model_type_str = model_type.value if hasattr(model_type, "value") else str(model_type)

    # Use scenario-aware evaluation path so scenario/non-scenario do not overwrite each other.
    url_file = get_hdt_eval_path(exp, model_type_str, "superprocess_url.txt", allow_simulation=allow_simulation)

    os.makedirs(os.path.dirname(url_file), exist_ok=True)
    with open(url_file, "w", encoding="utf-8") as f:
        f.write(log_url)
    return url_file


def _extract_instance_id(instance_url: str) -> str:
    """
    Extracts the numeric instance id from URLs like:
    https://.../flow/engine/158/  or  https://.../flow/engine/158
    """
    m = re.search(r"/flow/engine/(?P<id>\d+)/?$", instance_url)
    if not m:
        raise ValueError(f"Could not extract instance id from instance_url: {instance_url}")
    return m.group("id")


def _state_resource_url(instance_id: str) -> str:
    return f"https://{target_host}/flow/engine/{instance_id}/properties/state/"


def _uuid_resource_url(instance_id: str) -> str:
    return f"https://{target_host}/flow/engine/{instance_id}/properties/attributes/uuid/"


def _logs_url_from_uuid(uuid_value: str) -> str:
    uuid_value = uuid_value.strip()
    if not uuid_value:
        raise ValueError("UUID value is empty.")
    return f"https://{target_host}/logs/{uuid_value}.xes.yaml"


def _get_text(url: str, timeout_s: int = 30) -> str:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return (resp.text or "").strip()



def wait_until_finished_and_store_log_url(exp: str, model_type, instance_url: str, poll_seconds: int = 300, allow_simulation: bool = False):
    instance_id = _extract_instance_id(instance_url)
    state_url = _state_resource_url(instance_id)
    uuid_url = _uuid_resource_url(instance_id)

    print(f"Polling state every {poll_seconds}s: {state_url}")

    while True:
        state = _get_text(state_url).lower()
        print(f"Instance {instance_id} state: '{state}'")

        if state == "running":
            time.sleep(poll_seconds)
            continue

        if state == "finished":
            uuid_value = _get_text(uuid_url)
            log_url = _logs_url_from_uuid(uuid_value)
            stored_at = _store_superprocess_log_url(exp, model_type, log_url, allow_simulation=allow_simulation)
            print(f"Instance finished. Stored persistent log URL at: {stored_at}")
            return log_url

        if state == "stopped":
            print(f"Error: Instance {instance_id} entered state 'stopped'. Aborting polling.")
            return None

        print(f"State '{state}' is not 'running'/'finished'. Waiting {poll_seconds}s and retrying...")
        time.sleep(poll_seconds)



def run_predictive_simulation_and_extract(all_intermediate, bundle, fc_cfg, eval_cfg, cpee_predictive_simulation, worker_url, allow_simulation=False):
    # Init parameters

    cpee_configs = [list(tp) for tp, w in cpee_predictive_simulation.items()]

    model_type = fc_cfg.model_type
    exp = fc_cfg.exp
    component = Components.ALL
    scale_min = eval_cfg.scale_min
    scale_max = eval_cfg.scale_max

    best_models_dict = {}

    # Do we need a new superprocess execution?
    if not check_and_get_superprocess_log_url(exp, model_type, allow_simulation):
        # Input
        td, best_bundles = generate_superprocess_forecast_input_json(all_intermediate, bundle, cpee_predictive_simulation, fc_cfg)
        td[SCALE] = {SCALE_MIN: scale_min, SCALE_MAX: scale_max}
        input_file = f"forecast_input_{exp if not RUN_SCENARIO else get_scenario_exp(exp)}_{model_type.value}.json"
        input_path = f"{get_data_path(exp)}{input_file}"
        with open(input_path, "w") as outfile:
            json.dump(td, outfile, default=custom_json_serializer)

        upload_with_password(target_server, input_path, input_file, target_user, target_host)

        for hdt_config, (tp_cfg, w_bundle) in cpee_predictive_simulation.items():
            best, r = select_best_model_by_rmse(all_intermediate, tp_cfg, fc_cfg, allow_simulation=allow_simulation)
            if best:
                # Construct local path
                hdt_config_key = hdt_config[:3]  # (stride, horizon, input_size)
                best_models_dict[hdt_config_key] = best

                local_onnx_path = get_simple_model_path(
                    exp=exp,
                    mt=fc_cfg.model_type,
                    st_lt=fc_cfg.st_lt,
                    component=tp_cfg.component,
                    segmentation=best['seg'],
                    prediction_horizon=tp_cfg.prediction_horizon,
                    input_size=tp_cfg.input_size,
                    serialization_t="onnx",
                    mode=best['mode']
                )
                if os.path.exists(local_onnx_path):
                    # Construct remote name: [model_type, exp, component, horizon, input_size].join("_") + '.onnx'
                    remote_file_parts = [
                        model_type.value,
                        exp,
                        tp_cfg.component.value,
                        str(tp_cfg.prediction_horizon),
                        str(tp_cfg.input_size)
                    ]
                    sim_suffix = "_sim" if allow_simulation else ""
                    remote_filename = "_".join(remote_file_parts) + sim_suffix + ".onnx"

                    # Target directory: model_type + "_" + exp
                    remote_dir = f"{model_type.value}_{exp}"

                    print(f"Uploading best model: {local_onnx_path} to {remote_dir}/{remote_filename}")
                    upload_with_password(target_server + '/pred_models/' + remote_dir, local_onnx_path, remote_filename,
                                         target_user, target_host)
                else:
                    print(f"Warning: ONNX model not found at {local_onnx_path}")

        cpee_configs.reverse()  # longest running experiment at beginning
        cycles = [len(best_bundles[tp].chunks) for tp, d in cpee_predictive_simulation.items()]
        cycles.reverse()
        #start_modes = [td[horizon][stride][size][0][fc_cfg.mode_prefix] for stride, horizon, size, lower, upper in cpee_predictive_simulation.keys()]
        #start_modes.reverse()
        data_init = {
            "combined": cpee_configs,
            "exp": exp,
            "model_type": model_type.value,
            "component": "all",
            "remotesub": worker_url,
            "scale_min": eval_cfg.scale_min,
            "scale_max": eval_cfg.scale_max,
            "cycles": cycles,
            "sim": allow_simulation,
            #"start_modes": start_modes,
            "scenario_exp": exp if not RUN_SCENARIO else get_scenario_exp(exp)
            #"actuators": len(MODES[exp]),
            #"sensors": len(bundle.train_scaled.components)
        }
        attribute_init = {
            "executionhandler": "ruby"
        }

        json_payload = json.dumps(data_init, default=custom_json_serializer)
        init_arg = f"init={json_payload}"

        attribute_payload = json.dumps(attribute_init)
        attr_init_arg = f"attributes={attribute_payload}"

        cpee_call_string = ["curl",
                            f"https://{target_host}{target_start}",
                            "-X",
                            "POST",
                            "-F",
                            "behavior=fork_running",
                            "-F",
                            f"url=https://{target_host}{target_cpee}superprocess.xml",
                            "-F",
                            init_arg,
                            "-F",
                            attr_init_arg]
        print(cpee_call_string)
        no_retries = 3
        instance_url = None
        for i in range(no_retries):
            # Retries allowed
            try:
                superprocess = subprocess.run(cpee_call_string, capture_output=True, text=True)
                out_dict = json.loads(superprocess.stdout)
                print(out_dict)
                instance_id = out_dict[CPEE_INSTANCE]
                instance_url = out_dict[CPEE_INSTANCE_URL]
                print(f"Started CPEE instance with id {instance_id} and url {instance_url}.")
                break
            except json.decoder.JSONDecodeError:
                if i == no_retries - 1:
                    print("Cannot properly call the CPEE. Check!")
                else:
                    print("Error in Json decoding of the return string during calling the CPEE. Retrying...")
        # Wait until log can be retrieved
        if not instance_url:
            print("Failed to start CPEE instance: instance_url is missing.")
            return None, None, best_bundles, best_models_dict

        # Poll properties endpoint until finished, then store the constructed logs URL in superprocess_url.txt
        finished_log_url = wait_until_finished_and_store_log_url(
                exp=exp,
                model_type=model_type,
                instance_url=instance_url,
                poll_seconds=300,
                allow_simulation=allow_simulation
            )
        if finished_log_url is None:
            print(f"Error: Superprocess {fc_cfg} {cpee_configs} did not finish successfully; no log URL available.")
            return None, None, best_bundles
        log_url = check_and_get_superprocess_log_url(exp, model_type, allow_simulation,True)
    else:
        td, best_bundles = generate_superprocess_forecast_input_json(all_intermediate, bundle,
                                                                                  cpee_predictive_simulation, fc_cfg)
        log_url = check_and_get_superprocess_log_url(exp, model_type, allow_simulation,True)
        # Build best_models_dict for cached case too
        for hdt_config, (tp_cfg, w_bundle) in cpee_predictive_simulation.items():
            best_result = select_best_model_by_rmse(all_intermediate, tp_cfg, fc_cfg, allow_simulation=allow_simulation)
            if best_result:
                best, r = best_result
                hdt_config_key = hdt_config[:3]
                best_models_dict[hdt_config_key] = best

    hdt_runs, t = retrieve_superprocess_subprocesses(log_url, model_type, exp, allow_simulation)

    pred_futures = {}
    pred_modes = {}
    tp_cfgs = {}
    for i, (sid, sname) in enumerate(hdt_runs.items()):
        exp_sname = exp if not RUN_SCENARIO else get_scenario_exp(exp)
        if exp_sname == sname:
            #next_comb = cpee_configs[len(cpee_configs) - i - 1]
            next_comb = cpee_configs[i - 1]
            st = next_comb[0]
            h = next_comb[1]
            s = next_comb[2]
            #chunk_duration = pd.Timedelta(h * SAMPLING_FACTOR, "s")
            #stride_duration = pd.Timedelta(s * SAMPLING_FACTOR, "s")
            #chunks_test = chunk_series(bundle.test_scaled, chunk_duration, stride_duration)
            chunks_test = best_bundles[tuple(next_comb)].chunks
            log_name = get_hdt_eval_file(exp, model_type, get_log_name(sid), allow_simulation=allow_simulation)
            parsed_log_pkl, pred_futures_pkl, pred_modes_pkl = cache_paths(log_name, sid)
            if os.path.exists(parsed_log_pkl) and os.path.exists(pred_futures_pkl) and os.path.exists(pred_modes_pkl) and not RECOMPUTE_FUTURES:
                try:
                    cached_chunks = _load_pickle(pred_futures_pkl)
                    cached_modes = _load_pickle(pred_modes_pkl)
                    pred_futures[(st, h, s)] = cached_chunks
                    pred_modes[(st, h, s)] = cached_modes
                    continue
                except Exception as e:
                    print(f"Warning: cache load failed for sid={sid}, recomputing. Reason: {e}")
            if os.path.exists(parsed_log_pkl):
                log = _load_pickle(parsed_log_pkl)
            else:
                log = read_yaml(log_name, variant="full")
                log.drop(
                    columns=['concept:instance', 'id:id', 'cpee:activity', 'cpee:instance', 'lifecycle:transition'],
                    inplace=True)

            try:
                _save_pickle(parsed_log_pkl, log)
            except Exception as e:
                print(f"Warning: log cache save failed for sid={sid}. Reason: {e}")

            if exp in [HEATING, WATERLEVEL, IRRIGATION]:
                try:
                    df_sim = log[((log["concept:name"].isin(SIM_EXTRACTOR_TASKS[exp])) & (
                            log["cpee:lifecycle:transition"] == "dataelements/change"))]
                    df_sensor = log.loc[
                        log["concept:name"].astype("string").str.startswith(EXTRACTOR_TASK, na=False)
                        & (log["cpee:lifecycle:transition"] == "dataelements/change")
                        ].copy()
                    print(f"Sanity checking {fc_cfg} {sid} {next_comb}: Number of sensors equals number of test sensors {len(chunks_test) * h == len(df_sensor)}. Missing events {len(chunks_test) * h - len(df_sensor)}")
                    extracted = df_sensor["data"].apply(extract_sim_time_and_result)
                    df_sensor = df_sensor.join(extracted)

                    df_sensor["simulation_time"] = pd.to_numeric(df_sensor["simulation_time"], errors="coerce")
                    df_sensor["result"] = pd.to_numeric(df_sensor["result"], errors="coerce")
                    sensors = df_sensor["result"].to_list()
                    sim_times = df_sensor["simulation_time"].to_list()

                    break_indices = [0]
                    for i in range(1, len(sim_times)):
                        if sim_times[i] < sim_times[i - 1]:
                            break_indices.append(i)
                    break_indices.append(len(sim_times))

                    chunks = []
                    sensor_names = SENSOR_NAMES[exp]
                    for k in range(len(break_indices) - 1):
                        start = break_indices[k]
                        end = break_indices[k + 1]

                        # Use the timestamp from the start of the chunk as the base
                        base_time = chunks_test[k].start_time() - pd.Timedelta(1, "s")
                        end_time = chunks_test[k].end_time()

                        data = {"time": [base_time + pd.to_timedelta(s, unit='s') for s in sim_times[start:end]]}
                        if len(sensor_names) == 1:
                            # Single sensor case
                            data[sensor_names[0]] = sensors[start:end]
                        else:
                            # Multiple sensors case (if sensors were a list of lists)
                            sensor_data = list(zip(*sensors[start:end]))
                            for i, sensor_name in enumerate(sensor_names):
                                data[sensor_name] = sensor_data[i]

                        temp_df = pd.DataFrame(data)
                        #temp_df = temp_df.set_index("time")
                        temp_df = temp_df.set_index("time")
                        temp_df = temp_df.resample('1s').ffill()
                        # Create the Darts TimeSeries
                        ts = TimeSeries.from_dataframe(
                            temp_df,
                            value_cols=sensor_names,
                            freq='S'
                        )
                        ts = ts.drop_after(end_time, keep_point=True)

                        chunks.append(ts)
                    pred_futures[(int(st), int(h), int(s))] = chunks
                    pred_modes[(int(st), int(h), int(s))] =  extract_sim_tim_and_actuator_modes(df_sim, df_sensor, break_indices, chunks_test, sim_times, exp, best_bundles[tuple(next_comb)])

                except IndexError as e:
                    print(f"Extracting predictions failed for {fc_cfg} sid={sid}. Reason: {e}")
                try:
                    #_save_pickle(pred_futures_pkl, chunks)
                    _save_pickle(pred_modes_pkl, pred_modes)
                except Exception as e:
                    print(f"Warning: cache save failed for sid={sid}. Reason: {e}")

    return pred_futures, pred_modes, hdt_runs, best_bundles, best_models_dict


def _to_dict(cell):
    """Return a Python dict from either a dict-like object or a string literal."""
    if isinstance(cell, dict):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, str):
        s = cell.strip()
        # If the string contains escaped quotes like \', try unescaping first
        # (works if your string literally contains backslashes)
        if "\\'" in s or '\\"' in s:
            try:
                s = bytes(s, "utf-8").decode("unicode_escape")
            except Exception:
                pass
        return ast.literal_eval(s)
    # Unknown type
    return None


def extract_sim_tim_and_actuator_modes(df_sim, df_sensor, break_indices, chunks_test, sim_times, exp, w_bundle):
    #if ALLOW_SIMULATION:
    mode_name = list(w_bundle.all_past_covariates.components)[0]

    df_sim_copy = df_sim.copy()
    df_sim_copy["_encoded"] = df_sim_copy["concept:name"].apply(
        lambda cell: encode_modes(exp, cell, True)
    )

    # df_sensor already has simulation_time; align df_sim rows to simulation time
    # by finding the closest preceding df_sensor row by real timestamp
    sensor_ts_sorted = df_sensor.sort_values("time:timestamp")
    sim_ts_sorted = df_sim_copy.sort_values("time:timestamp")

    # Use merge_asof to get the simulation_time of the last sensor row
    # that was recorded at or before each df_sim row's real timestamp
    merged_sim = pd.merge_asof(
        sim_ts_sorted[["time:timestamp", "_encoded"]].reset_index(drop=True),
        sensor_ts_sorted[["time:timestamp", "simulation_time"]].reset_index(drop=True),
        on="time:timestamp",
        direction="backward"
    )
    # Drop rows where no preceding sensor event was found
    merged_sim = merged_sim.dropna(subset=["simulation_time"])
    merged_sim["simulation_time"] = pd.to_numeric(merged_sim["simulation_time"], errors="coerce")
    merged_sim = merged_sim.dropna(subset=["simulation_time"])

    mode_sim_times = merged_sim["simulation_time"].to_list()
    mode_values = merged_sim["_encoded"].to_list()

    # Determine break indices for mode chunks (same logic as for sensors)
    mode_break_indices = [0]
    for i in range(1, len(mode_sim_times)):
        if mode_sim_times[i] < mode_sim_times[i - 1]:
            mode_break_indices.append(i)
    mode_break_indices.append(len(mode_sim_times))

    pred_modes_chunks = []
    for k in range(len(break_indices) - 1):
        base_time = chunks_test[k].start_time() - pd.Timedelta(1, "s")
        end_time = chunks_test[k].end_time()

        # Get the sim_times for this chunk to build the full timeline
        chunk_start_idx = break_indices[k]
        chunk_end_idx = break_indices[k + 1]
        chunk_sim_times_all = sim_times[chunk_start_idx:chunk_end_idx]

        # Get mode events for this chunk
        if k < len(mode_break_indices) - 1:
            m_start = mode_break_indices[k]
            m_end = mode_break_indices[k + 1]
            chunk_mode_times = mode_sim_times[m_start:m_end]
            chunk_mode_values = mode_values[m_start:m_end]
        else:
            chunk_mode_times = []
            chunk_mode_values = []

        # Build a mapping: sim_time -> mode_value with forward fill logic
        mode_dict = {}

        if chunk_mode_times:
            # Determine initial mode (opposite of first recorded mode)
            if chunk_mode_times[0] != 1:
                first_mode_time = chunk_mode_times[0]
                first_mode_value = chunk_mode_values[0]
                initial_mode = 1 - first_mode_value  # Flip: 0->1, 1->0
            else:
                first_mode_time = 1
                initial_mode = chunk_mode_values[0]

            # Fill all sim times before the first mode change with the opposite value
            for simt in chunk_sim_times_all:
                if simt < first_mode_time:
                    mode_dict[simt] = initial_mode
                else:
                    break

            # forward-fill from each mode change point
            for i, (mode_time, mode_val) in enumerate(zip(chunk_mode_times, chunk_mode_values)):
                # Determine the range this mode value covers
                next_mode_time = chunk_mode_times[i + 1] if i + 1 < len(chunk_mode_times) else float(
                    'inf')

                # Fill all sim_times from mode_time up to (but not including) next_mode_time
                for simt in chunk_sim_times_all:
                    if mode_time <= simt < next_mode_time:
                        mode_dict[simt] = mode_val
        else:
            # No mode events recorded for this chunk
            # Fill with NaN or use a default (e.g., 0)
            for simt in chunk_sim_times_all:
                mode_dict[simt] = float("nan")

        # Now create the DataFrame with forward-filled values
        mode_df = pd.DataFrame({
            "time": [base_time + pd.to_timedelta(simt, unit='s') for simt in chunk_sim_times_all],
            mode_name: [mode_dict.get(simt, float("nan")) for simt in chunk_sim_times_all]
        }).set_index("time")

        # mode_df = mode_df[~mode_df.index.duplicated(keep='last')]

        # mode_df = mode_df.resample('1s').ffill().bfill()

        mode_df.index.name = "time"
        mode_ts = TimeSeries.from_dataframe(
            mode_df,
            value_cols=[mode_name],
            freq="S"
        )
        # mode_ts = mode_ts.drop_after(end_time, keep_point=True)
        pred_modes_chunks.append(mode_ts)
    return pred_modes_chunks

def extract_sim_time_and_result(cell):
    d = _to_dict(cell)
    if not d or "children" not in d:
        return pd.Series({"simulation_time": pd.NA, "result": pd.NA})
    sim_time = pd.NA
    result = pd.NA
    for _key, node in d.get("children", []):
        inner = (node or {}).get("children", {})
        name = inner.get("name")
        value = inner.get("value")
        if name == "simulation_time":
            sim_time = value
        elif name == "result":
            # Handle multiple possible formats for the result value:

            # Format 1: Direct list [0.2023553]
            if isinstance(value, list) and value:
                result = value[0]
            # Format 2: Nested dict with children: {'value': None, 'children': [0.018819999]}
            elif isinstance(value, dict):
                if "children" in value:
                    children = value.get("children", [])
                    if isinstance(children, list) and children:
                        result = children[0]
                    else:
                        # Fallback to the value itself if children is empty
                        result = value.get("value")
                else:
                    # Dict without 'children' key
                    result = value.get("value", value)
            # Format 3: Direct value (number, string, etc.)
            else:
                result = value
    return pd.Series({"simulation_time": sim_time, "result": result})




def prepare_scenario(cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg):
    # update data bundle
    # update worker bundles past covariates
    some_tp_cfg = cpee_predictive_simulation[list(cpee_predictive_simulation.keys())[0]][0]
    data_bundle = load_or_prepare_scenario_bundle(forecast_config, eval_cfg, data_bundle, some_tp_cfg.fold)
    exp = forecast_config.exp
    new_cpee_predictive_simulation = {}
    for comb, (tp_cfg, w_bundles) in cpee_predictive_simulation.items():
        chunks = construct_sensor_chunks(data_bundle, forecast_config, tp_cfg)
        new_w_bundles = []
        for i in range(len(w_bundles)):
            if i == 0:
                # Non-simulated control modes
                new_w_bundle = construct_worker_bundle(data_bundle, chunks, get_all_mode_names(exp))
                new_w_bundles.append(new_w_bundle)
            elif i == 1:
                sim_mode_component = get_all_sim_mode_components(exp)
                last_control = w_bundles[0].past_covariates[get_all_mode_names(exp)].first_value()
                all_past_covariates_sim = build_ts_sim_from_covariates(data_bundle, last_control, tp_cfg.lower, tp_cfg.upper,
                                                                       forecast_config,
                                                                       eval_cfg.scale_min, eval_cfg.scale_max,
                                                                       sim_mode_component)
                past_covariates_sim = all_past_covariates_sim.drop_after(data_bundle.val_scaled.start_time())
                val_past_covariates_sim = all_past_covariates_sim.drop_after(data_bundle.val_scaled.end_time(),
                                                                             keep_point=True).drop_before(
                    data_bundle.train_scaled.end_time())
                new_w_bundle = WorkerBundle(chunks, past_covariates_sim, val_past_covariates_sim, all_past_covariates_sim)
                new_w_bundles.append(new_w_bundle)
        new_cpee_predictive_simulation[comb] = (tp_cfg, new_w_bundles)
    return new_cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg


def cache_dir_for_logfile(log_file: str) -> str:
    d = os.path.join(os.path.dirname(log_file), "parsed_cache")
    os.makedirs(d, exist_ok=True)
    return d

def cache_paths(log_file: str, sid: str) -> tuple[str, str, str]:
    cache_dir = cache_dir_for_logfile(log_file)
    parsed_log_pkl = os.path.join(cache_dir, f"{sid}_parsed_log.pkl")
    pred_futures_pkl = os.path.join(cache_dir, f"{sid}_pred_futures.pkl")
    pred_modes_pkl = os.path.join(cache_dir, f"{sid}_pred_modes.pkl")
    return parsed_log_pkl, pred_futures_pkl, pred_modes_pkl
