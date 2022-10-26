echo "Let's start build database!"
python -m src build_database configs/rencos_python_base12.yaml \
              --ckpt=models/rencos_python/transformer_base12/401683.ckpt \
              --hidden_representation_path=saved/datastore_401683/embedding \
              --token_map_path=saved/datastore_401683/token_map \
              --index_path=saved/datastore_401683/index