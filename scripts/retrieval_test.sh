echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=0
python -m src retrieval_test configs/rencos_java_base2.yaml \
        --ckpt saved/datastore_java/base2/433621.ckpt