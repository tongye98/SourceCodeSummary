# coding: utf-8
"""
Evaluation metrics
"""
import logging 
import collections
import math
logger = logging.getLogger(__name__)


class Bleu(object):
    """
    Python implementation of BLEU and smooth-BLEU.
    This module provides a Python implementation of BLEU and smooth-BLEU.
    Smooth BLEU is computed following the method outlined in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    def __init__(self) -> None:
        pass
    def _get_ngrams(self,segment, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.
        Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts


    def compute_bleu(self, reference_corpus, translation_corpus, max_order=4,
                    smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(reference_corpus,
                                            translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, max_order)
            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                    possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        # return (bleu, precisions, bp, ratio, translation_length, reference_length)
        bleu_order = {}
        for i, precision in enumerate(precisions):
            bleu_order[i+1] = precision * bp
        return bleu, bleu_order


    def corpus_bleu(self, hypotheses, references):
        """
        From NeualCodeSum google-bleu.
        """
        refs = []
        hyps = []
        count = 0
        total_score = 0.0

        Ids = len(references)
        ind_score = dict()

        for id in range(Ids):
            hyp = hypotheses[id].split()    # ['partitions', 'a', 'list', 'of', 'suite', 'from', 'a', 'interval', '.']
            ref = [references[id].split()]  # [['reorders', 'a', 'test', 'suite', 'by', 'test', 'type', '.']]
            hyps.append(hyp)
            refs.append(ref)

            score = self.compute_bleu([ref], [hyp], smooth=True)[0]
            total_score += score
            count += 1
            ind_score[id] = score

        avg_score = total_score / count
        corpus_bleu, bleu_order = self.compute_bleu(refs, hyps, smooth=True)
        # return corpus_bleu, avg_score, ind_score
        return corpus_bleu, bleu_order

class Rouge_l(object):
    pass 

class Meteor(object):
    pass 





if __name__ == "__main__":
    predictions_path = "test/predictions"
    references_path = "test/references"
    with open(predictions_path,"r") as p, open(references_path, "r") as r:
        predictions = p.read().splitlines()  # list of string/sentence
        references = r.read().splitlines()  # list of string/sentence
        assert len(predictions) == len(references)

        bleu = Bleu()
        corpus_bleu, bleu_order = bleu.corpus_bleu(hypotheses=predictions, references=references)
        print("corpus_bleu : {}".format(corpus_bleu))
        print("Bleu order1-4 :",bleu_order)
        # corpus_bleu : 0.25467977003051817
        # {1: 0.45147210394009457, 2: 0.25286423679476816, 3: 0.20369564445128535, 4: 0.18091631042057468}