echo "Let's start retrieval test!"
export CUDA_VISIBLE_DEVICES=1
python -m src retrieval_test configs/codescribe_java.yaml \
        --ckpt  models/codescribe_java/209184.ckpt