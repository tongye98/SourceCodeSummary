# use rencos method.
echo "Let's start check rencos method!"
export CUDA_VISIBLE_DEVICES=1
python -m src rencos_test configs/rencos_python_base12.yaml \
        --ckpt models/rencos_python/transformer_base12/401683.ckpt