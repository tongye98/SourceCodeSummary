echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=1
python -m src retrieval_test configs/rencos_python_base12.yaml \
        --ckpt models/rencos_python/transformer_base12/401683.ckpt