echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=1
python -m src build_code_semantic datastore/datastore_liu_c/base3/code_inner/liu_c_base3.yaml