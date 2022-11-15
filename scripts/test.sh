#! /bin/bash
# bash scripts/test.sh 1 test configs/rencos_python_base1.yaml

echo "Let's testing the model"
export CUDA_VISIBLE_DEVICES=0
python -m src test configs/leclair_java_base5.yaml --ckpt models/leclair_java/transformer_base5/best.ckpt