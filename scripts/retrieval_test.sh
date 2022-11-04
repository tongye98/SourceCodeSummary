echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=1
python -m src retrieval_test configs/rencos_java_base4.yaml \
        --ckpt saved/datastore_java/base4/431442.ckpt