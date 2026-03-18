import darts.metrics
import pandas as pd

from hdt.remote_util import *
from darts.metrics import accuracy
import subprocess
import json
import pm4py
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque


from hdt.util import select_best_model_by_rmse, custom_json_serializer
from hdt.run.run import evaluate
from hdt.run.hdt_evaluation import run_predictive_simulation_and_extract, prepare_scenario
from hdt.run.configuration import SeriesBundle


def run_predictive_simulation_and_extract_parallel(model_jobs, all_intermediate, workers, allow_simulation=False):
    """
    Parallelized execution across multiple workers.
    Submits jobs to workers and processes results as they complete.
    """
    # Queue to manage work distribution
    work_queue = deque()
    worker_pool = deque(workers)  # Available workers

    # Global dict to aggregate worker results
    global_results = {}
    # Lock for thread-safe access to global_results
    results_lock = threading.Lock()

    # Process each model job
    for model, exp, st_lt, fold, model_assignment in model_jobs:
        # Execute the model to gather inputs
        result = model_assignment(exp, st_lt, all_intermediate, fold)

        if result is None:
            continue

        cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg = result
        if DIFFERENCE_TS in data_bundle.train_scaled.components:
            data_bundle = SeriesBundle(train_scaled=data_bundle.train_scaled.drop_columns(DIFFERENCE_TS),
                                       val_scaled=data_bundle.val_scaled.drop_columns(DIFFERENCE_TS),
                                       test_scaled=data_bundle.test_scaled.drop_columns(DIFFERENCE_TS))
        if RUN_SCENARIO and st_lt is Horizon.ST:
            cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg = prepare_scenario(
                cpee_predictive_simulation, data_bundle, forecast_config, eval_cfg)
        if len(cpee_predictive_simulation) > 0 and not SKIP_CPEE:
            work_queue.append({
                'model': model,
                'exp': exp,
                'st_lt': st_lt,
                'fold': fold,
                'cpee_predictive_simulation': cpee_predictive_simulation,
                'data_bundle': data_bundle,
                'forecast_config': forecast_config,
                'eval_cfg': eval_cfg,
                'all_intermediate': all_intermediate,
                'allow_simulation': allow_simulation
            })

    # Process work queue with worker pool
    active_jobs = {}
    completed_count = 0
    total_jobs = len(work_queue)

    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        # Initial submission - fill all workers
        while work_queue and worker_pool:
            job = work_queue.popleft()
            worker = worker_pool.popleft()

            future = executor.submit(
                run_single_model_on_worker,
                job['all_intermediate'],
                job['data_bundle'],
                job['forecast_config'],
                job['eval_cfg'],
                job['cpee_predictive_simulation'],
                worker,
                job['allow_simulation']
            )
            active_jobs[future] = (worker, job)
            print(f"Submitted {job['model'].value} for {job['exp']} to worker {worker}")

        # Process completions and submit new work
        while active_jobs:
            # Wait for any job to complete
            for future in as_completed(active_jobs):
                worker, job = active_jobs.pop(future)

                try:
                    pred_futures, pred_modes, hdt_runs, best_bundles, best_models_dict = future.result()
                    completed_count += 1

                    print(f"Completed {job['model'].value} for {job['exp']} ({completed_count}/{total_jobs})")

                    # Worker-specific results dict
                    worker_results = {}

                    # Evaluate results
                    for hdt_configuration in pred_futures:
                        next_comb = next(
                            (t for t in job['cpee_predictive_simulation'].keys() if t[:3] == hdt_configuration), None)
                        tp_cfg = job['cpee_predictive_simulation'][next_comb][0]
                        mode_component = get_all_mode_names(job['forecast_config'].exp)
                        if any([m in best_bundles[next_comb].chunks[0].columns for m in mode_component]):
                            chunks_test = [ best_bundles[next_comb].chunks[i].drop_columns(mode_component) for i in range(len(best_bundles[next_comb].chunks))]
                        else:
                            chunks_test = best_bundles[next_comb].chunks
                        test_covariates = best_bundles[next_comb].all_past_covariates.drop_before(chunks_test[0].start_time(), keep_point=True)
                        test_covariates_chunks = chunk_series(test_covariates, pd.Timedelta(tp_cfg.input_size * SAMPLING_FACTOR, "s"),
                                                              pd.Timedelta(tp_cfg.prediction_stride * SAMPLING_FACTOR, "s"))
                        print(f"Scenario: {RUN_SCENARIO}")
                        tp_cfg = replace(tp_cfg, segmentation=tp_cfg.component if not RUN_SCENARIO else Components.SENSORS_ONLY)
                        avgs = quick_evaluation(
                            chunks_test,
                            pred_futures[hdt_configuration],
                            job['forecast_config'],
                            tp_cfg,
                        )
                        accuracy = quick_accuracy(test_covariates_chunks, pred_modes[hdt_configuration], job['forecast_config'], tp_cfg)
                        print(
                            f"fc_cfg: {job['forecast_config']} tp_cfg: {tp_cfg} for configuration {next_comb} have resulted in the following evaluation: {avgs}")
                        print(
                            f"fc_cfg: {job['forecast_config']} tp_cfg: {tp_cfg} for configuration {next_comb} have resulted in the following actuator mode accuracy: {accuracy}")
                        # Get best model info for this configuration
                        best_model_info = best_models_dict.get(hdt_configuration, {})

                        # Store worker results: avgs[0]=WAPE, avgs[1]=RMSE, avgs[2]=MAE
                        # model and exp must be additionally keys
                        config_key = next_comb + (job['model'], job['exp'], job['fold'])
                        worker_results[config_key] = {
                            'scores': {
                                'WAPE': avgs[0],
                                'RMSE': avgs[1],
                                'MAE': avgs[2]
                            },
                            'job_context': {
                                'exp': job['exp'],
                                'model_type': job['forecast_config'].model_type,
                                'st_lt': job['st_lt'],
                                'component': tp_cfg.component,
                                'segmentation': best_model_info.get('seg', tp_cfg.segmentation),
                                'prediction_horizon': tp_cfg.prediction_horizon,
                                'prediction_stride': tp_cfg.prediction_stride,
                                'input_size': tp_cfg.input_size,
                                'multi_mode': best_model_info.get('mode', None)
                            }
                        }

                    # Aggregate worker results into global dict (thread-safe)
                    with results_lock:
                        if worker not in global_results:
                            global_results[worker] = {}
                        for config, data in worker_results.items():
                            global_results[worker][config] = data

                except Exception as e:
                    print(f"Error processing {job['model'].value} for {job['exp']}: {e}")
                    import traceback
                    traceback.print_exc()

                # Worker is now free - assign new work if available
                if work_queue:
                    next_job = work_queue.popleft()
                    new_future = executor.submit(
                        run_single_model_on_worker,
                        next_job['all_intermediate'],
                        next_job['data_bundle'],
                        next_job['forecast_config'],
                        next_job['eval_cfg'],
                        next_job['cpee_predictive_simulation'],
                        worker,
                        next_job['allow_simulation'],
                    )
                    active_jobs[new_future] = (worker, next_job)
                    print(f"Submitted {next_job['model'].value} for {next_job['exp']} to worker {worker}")
                else:
                    # No more work, add worker back to pool
                    worker_pool.append(worker)

                break  # Exit the for loop to restart as_completed

    # After all jobs are completed, populate the intermediate structure
    _populate_intermediate_from_global_results(global_results, all_intermediate)

    return global_results


def _populate_intermediate_from_global_results(global_results, all_intermediate):
    """
    Populate the intermediate structure with results from global_results dict.

    Args:
        global_results: Dict[worker_url, Dict[config_tuple, Dict['scores', 'job_context']]]
        all_intermediate: The intermediate structure to populate
    """
    from hdt.parameters import Scores
    from hdt.util import assign_intermediate, store_intermediate

    print("\n" + "=" * 80)
    print("Populating intermediate structure with aggregated results")
    print("=" * 80)

    for worker_url, worker_data in global_results.items():
        print(f"\nProcessing results from worker: {worker_url}")

        for config_tuple, data in worker_data.items():
            scores = data['scores']
            ctx = data['job_context']

            print(f"  Config: stride={ctx['prediction_stride']}, horizon={ctx['prediction_horizon']}, "
                  f"input_size={ctx['input_size']}, mode={ctx['multi_mode']}, seg={ctx['segmentation']}")
            print(f"  Scores: WAPE={scores['WAPE']:.4f}, RMSE={scores['RMSE']:.4f}, MAE={scores['MAE']:.4f}")

            # Populate each score in the intermediate structure
            for score_name in ['WAPE', 'RMSE', 'MAE']:
                assign_intermediate(
                    all_intermediate,
                    ctx['exp'],
                    ctx['model_type'],
                    ctx['st_lt'],
                    ctx['segmentation'],
                    ctx['component'] if not RUN_SCENARIO else Components.SENSORS_ONLY,
                    [scores[score_name]],  # Wrap in list as expected by assign_intermediate
                    ctx['prediction_horizon'],
                    ctx['prediction_stride'],
                    ctx['input_size'],
                    score_key=score_name,
                    mode=ctx['multi_mode']
                )
    if ADD_EVAL_VALUES_TO_FINAL_RESULTS:
        store_intermediate(all_intermediate, eval=True)
    # Store the updated intermediate structure
    #store_intermediate(all_intermediate)


def run_single_model_on_worker(all_intermediate, bundle, fc_cfg, eval_cfg, cpee_predictive_simulation, worker_url, allow_simulation=False):

    pred_futures, pred_modes, hdt_runs, best_bundles, best_models_dict = run_predictive_simulation_and_extract(all_intermediate, bundle, fc_cfg, eval_cfg, cpee_predictive_simulation,
                                          worker_url, allow_simulation=allow_simulation)
    return pred_futures, pred_modes, hdt_runs, best_bundles, best_models_dict

