echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=1
python -m src build_code_semantic datastore/datastore_liu_ccsd/transformer_base/code_inner/liu_ccsd_base1.yaml