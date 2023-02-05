#! /bin/bash
# bash scripts/test.sh 1 test configs/rencos_python_base1.yaml

echo "Let's testing the model"
export CUDA_VISIBLE_DEVICES=1
python -m src test configs/codescribe_python_base1.yaml --ckpt models/codescribe_python/transformer_base1/best.ckpt