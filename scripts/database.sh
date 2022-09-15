echo "Let's start build database!"
echo $0
export CUDA_VISIBLE_DEVICES=$1
python -m src build_database configs/rencos_python_base3.yaml \
              --ckpt=models/rencos_python/transformer_base3/best.ckpt \
              --embedding_path=test/embedding \
              --token_map_path=test/token_map \
              --index_path=test/index