#! /bin/bash
# bash scripts/test.sh 1 test configs/rencos_python_base1.yaml

echo "Let's testing the model"
export CUDA_VISIBLE_DEVICES=0
python -m src test configs/rencos_java_base4.yaml --ckpt models/rencos_java/transformer_base4/best.ckpt