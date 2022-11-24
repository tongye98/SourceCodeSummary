echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=1
python -m src retrieval_test configs/ahmad_java_base2.yaml \
        --ckpt datastore/datastore_ahmad_java/416189.ckpt