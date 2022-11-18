echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=0
python -m src retrieval_test configs/rencos_python_base12.yaml \
        --ckpt datastore/datastore_rencos_python/transformer_base12/401683.ckpt