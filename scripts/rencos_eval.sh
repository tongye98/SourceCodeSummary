echo "Let's use rencos metrics!"
python src/rencos_evaluation/evaluate.py \
models/rencos_python/transformer_base12_static/output_static_retrieval_inner_03_analysis_beam_mx=0.5bandwidth=20topk=16 \
data/rencos_python/test.summary \
50