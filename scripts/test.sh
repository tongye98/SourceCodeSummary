#! /bin/bash
# bash scripts/test.sh 1 test configs/rencos_python_base1.yaml

echo "Let's testing the model"
export CUDA_VISIBLE_DEVICES=0
# python -m src test $2 --ckpt $3 --output_path $4
python -m src test configs/rencos_python_base4.yaml --ckpt models/rencos_python/transformer_base4/best.ckpt --output_path models/rencos_python/transformer_base4/out