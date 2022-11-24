from src.eval.bleu import corpus_bleu
from src.eval.rouge import Rouge
from src.eval.meteor import Meteor

def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    return bleu * 100, rouge_l * 100, meteor * 100

def test_metrics():
    predictions_path = "data/rencos_python/rencos.out"
    references_path = "data/rencos_python/test.summary"
    with open(predictions_path,"r") as p, open(references_path, "r") as r:
        predictions = p.read().splitlines()  # list of string/sentence
        references = r.read().splitlines()  # list of string/sentence
        assert len(predictions) == len(references)

        predictions_dict = {k: [v.strip().lower()] for k,v in enumerate(predictions)}
        # 0: ['partitions a list of suite from a interval .']
        references_dict = {k: [v.strip().lower()] for k,v in enumerate(references)}
        # 0: ['partitions a list of suite from a interval .']

        bleu, rouge_l, meteor = eval_accuracies(hypotheses=predictions_dict, references=references_dict)
        print("bleu = {}".format(bleu))
        print("rouge-l = {}".format(rouge_l))
        print("meteor = {}".format(meteor))

