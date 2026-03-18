# HyProTwin Implementation and Evaluation

This repository contains the code for the BPM submission "HyProTwin: Hybrid Digital Process Twins for
Predictive Simulation of Continuous Processes". 


## Structure

- hdt/ contains all source code for running HyProTwin and the evaluation 
- models/ contains all supervisory ProTwins per continuous process as part of a HyProTwin. the *_scenario models are for predictive simulation, the remaining models for short-term prediction. The superprocess CPEE process model is for orchestrating the evaluation with respect to HyProTwins. 
- data/ contains the datasets of continuous processes used in the evaluation. For each continuous process, the scenario subdirectory contains the future what-if scenario time series.
- tmp/ contains the final results json, intermediate results json, and logs from running the pipeline.py for training and predicting.
- lib/prediction-service/ contains the forecasting service as a RUST ONNX web service 


## Quick evaluation

**Prerequisites**: Python 3.13

**Tested on**: Fedora 43

```shell
chmod +x install.sh
./install.sh
source .venv/bin/activate
python aggregate_evaluation.py
```

Observe the Latex output table (Table 3 in the paper).
The log accuracies.out contains the average accuracies of actuator modes. 

## Long evaluation
Without GPU acceleration (e.g., GeForce RTX 4090), the evaluation likely takes months.

### Installation


```shell
chmod +x install.sh
./install.sh
source .venv/bin/activate
```

### Training and Predicting
```shell
python -m hdt.pipeline
```

### Evaluating
```shell
python -m hdt.cpee_eval_pipeline
```

**Remark on available files**:
So far, we can provide all intermediate results json files and logs in tmp/. Due to the large size of all evaluation files that exceeds 85GB (including trained models, predicted DXES event logs), we cannot provide all remaining files and, at the same time, meet the double blind requirements.
As soon as the double blind does not take effect anymore, we will provide all files in a suitable file storage for full transparency. 

**Remark on HyProTwin**:
For HyProTwin to fully function, you must install a CPEE server (download and install as described at www.cpee.org).
Next, configure the target_host to your server in hdt/config.yaml and upload the CPEE models in models/ to your server's /var/www/cpee_models/ . You must have ssh access to your server. If you have an ssh key with with a password, set environment variable SSH_PASSWORD to your password. 
Lastly, set the SKIP_CPEE constant to True in hdt/parameters.py. 
Run the RUST ONNX web service also on your CPEE server. 
Copy the forecast_input_exp.php data service to the /var/www/ of your CPEE server.
By executing the step as described in evaluating, HyProTwin is automatically executed on your CPEE server. 