#!/bin/sh
python3.13 -m venv .venv
source .venv/bin/activate

pip install darts lightgbm skl2onnx
pip install xes-yaml-pm4py-extension/pm4py-2.7.11.11-py3-none-any.whl
pip install statsforecast xgboost onnxruntime onnxmltools paramiko 
pip install neuralforecast onnxscript seaborn

