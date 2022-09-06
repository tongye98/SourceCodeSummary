#! /bin/bash

echo "Let's start do something interesting!"
export CUDA_VISIBLE_DEVICES=1
python -m src train configs/transformer.yaml