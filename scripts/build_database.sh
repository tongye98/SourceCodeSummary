echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=0
python -m src build_database configs/rencos_java_base4.yaml \
              --ckpt=saved/datastore_java/base4/431442.ckpt \
              --hidden_representation_path=saved/datastore_java/base4/inner/embedding \
              --token_map_path=saved/datastore_java/base4/inner/token_map \
              --index_path=saved/datastore_java/base4/inner/index \
              --data_dtype=float32