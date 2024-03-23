#! /bin/bash

echo "Let's testing the model"
export CUDA_VISIBLE_DEVICES=1
python -m src test configs/codescribe_python.yaml --ckpt models/codescribe_python/transformer_base/best.ckpt