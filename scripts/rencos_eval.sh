echo "Let's use rencos metrics!"
python src/rencos_evaluation/evaluate.py \
saved/transformer_base12/datastore_401683/inner3/output_analysis_beam_mx=0.6bandwidth=20topk=32 \
data/rencos_python/test.summary \
30