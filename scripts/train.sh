#! /bin/bash

echo "Let's start do something interesting!"
export CUDA_VISIBLE_DEVICES=0
python -m src train configs/rencos_python_base4.yaml