echo "Let's start build database!"
export CUDA_VISIBLE_DEVICES=0
python -m src build_database configs/rencos_java_base2.yaml \
              --ckpt=saved/datastore_java/base2/433621.ckpt \
              --hidden_representation_path=saved/datastore_java/base2/l2/embedding \
              --token_map_path=saved/datastore_java/base2/l2/token_map \
              --index_path=saved/datastore_java/base2/l2/index \
              --data_dtype=float32