echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=0
python -m src build_database configs/rencos_python_base12.yaml \
              --ckpt=datastore/datastore_rencos_python/transformer_base12/401683.ckpt \
              --hidden_representation_path=datastore/datastore_rencos_python/transformer_base12/datastore_401683/inner_attention_encode/embedding \
              --token_map_path=datastore/datastore_rencos_python/transformer_base12/datastore_401683/inner_attention_encode/token_map \
              --index_path=datastore/datastore_rencos_python/transformer_base12/datastore_401683/inner_attention_encode/index \
              --data_dtype=float32