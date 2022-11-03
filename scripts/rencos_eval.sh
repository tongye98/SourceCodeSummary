echo "Let's use rencos metrics!"
python src/rencos_evaluation/evaluate.py \
models/rencos_java/transformer_base4_static/output_static_retrieval_inner_02_beam_mx=0.6bandwidth=30topk=16 \
data/rencos_java/test.summary \
50