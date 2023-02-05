#! /bin/bash
# bash scripts/train.sh 0 train configs/rencos_python_base1.yaml

echo "Let's start do something interesting!"
export CUDA_VISIBLE_DEVICES=1
python -m src train configs/codescribe_python_base1.yaml
