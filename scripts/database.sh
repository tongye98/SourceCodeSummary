echo "Let's start build database!"
python -m src build_database configs/rencos_python_base8.yaml \
              --ckpt=models/rencos_python/transformer_base8/best.ckpt \
              --hidden_representation_path=test/embedding \
              --token_map_path=test/token_map \
              --index_path=test/index