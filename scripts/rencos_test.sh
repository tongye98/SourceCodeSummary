# use rencos method.
echo "Let's start check rencos method!"
export CUDA_VISIBLE_DEVICES=1
python -m src rencos_test models/rencos_python/repeat2/config.yaml \
        --ckpt models/rencos_python/repeat2/best.ckpt

# python -m src rencos_test configs/rencos_python_base12.yaml \
#         --ckpt datastore/datastore_rencos_python/transformer_base12/401683.ckpt