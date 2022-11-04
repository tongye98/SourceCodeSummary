# use rencos method.
echo "Let's start check rencos method!"
export CUDA_VISIBLE_DEVICES=1
python -m src rencos_test configs/rencos_java_base4.yaml \
        --ckpt saved/datastore_java/base4/431442.ckpt