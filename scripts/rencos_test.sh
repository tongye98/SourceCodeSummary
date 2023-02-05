# use rencos method.
echo "Let's start check rencos method!"
export CUDA_VISIBLE_DEVICES=1
python -m src rencos_test datastore/datastore_liu_c/base3/code_inner/liu_c_base3.yaml \
        --ckpt datastore/datastore_liu_c/base3/code_inner/516460.ckpt

# python -m src rencos_test configs/rencos_python_base12.yaml \
#         --ckpt datastore/datastore_rencos_python/transformer_base12/401683.ckpt