echo "Let's use rencos metrics!"
python src/rencos_evaluation/evaluate.py \
models/rencos_python/transformer_base12_static/output_static_retrieval_inner_mx=0.4bandwidth=100topk=8 \
data/rencos_python/test.summary \
50