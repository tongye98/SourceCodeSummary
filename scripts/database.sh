echo "Let's start build database!"
python -m src build_database saved/transformer_base12/rencos_python_base12_static_retrieval.yaml \
              --ckpt=saved/transformer_base12/401683.ckpt \
              --hidden_representation_path=saved/transformer_base12/datastore_401683/inner/embedding \
              --token_map_path=saved/transformer_base12/datastore_401683/inner/token_map \
              --index_path=saved/transformer_base12/datastore_401683/inner/index