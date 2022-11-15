echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=1
python -m src build_database configs/rencos_python_base12.yaml \
              --ckpt=saved/transformer_base12/401683.ckpt \
              --hidden_representation_path=saved/transformer_base12/datastore_401683/inner3/embedding \
              --token_map_path=saved/transformer_base12/datastore_401683/inner3/token_map \
              --index_path=saved/transformer_base12/datastore_401683/inner3/index \
              --data_dtype=float32