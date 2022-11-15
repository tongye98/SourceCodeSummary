# coding: utf-8
"""
Evaluation metrics
"""
import logging 
import collections
import math
import atexit
import logging
import os
import subprocess
import sys
import threading
import psutil
import numpy as np 

logger = logging.getLogger(__name__)

class Bleu(object):
    """
    Python implementation of BLEU and smooth-BLEU.
    This module provides a Python implementation of BLEU and smooth-BLEU.
    Smooth BLEU is computed following the method outlined in the paper:
    Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
    evaluation metrics for machine translation. COLING 2004.
    """
    def __init__(self):
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

        try:
            if ratio > 1.0:
                bp = 1.
            else:
                bp = math.exp(1 - 1. / ratio)
        except:
            bp = 0.
            logger.warning("In bleu: float division by zero.")
            logger.info("translation_length = {}".format(translation_length))
            logger.info("reference_length = {}".format(reference_length))

        bleu = geo_mean * bp

        # return (bleu, precisions, bp, ratio, translation_length, reference_length)
        bleu_order = {}
        for i, precision in enumerate(precisions):
            bleu_order["bleu-{}".format(i+1)] = precision * bp
        return bleu, bleu_order

    def corpus_bleu(self, hypotheses, references):
        """
        From NeualCodeSum google-bleu.
        """
        refs = []
        hyps = []
        count = 0
        total_score = 0.0

        assert sorted(hypotheses.keys()) == sorted(references.keys())
        Ids = list(references.keys())
        ind_score = dict()

        for id in Ids:
            hyp = hypotheses[id][0].split()    # ['partitions', 'a', 'list', 'of', 'suite', 'from', 'a', 'interval', '.']
            ref = [references[id][0].split()]  # [['reorders', 'a', 'test', 'suite', 'by', 'test', 'type', '.']]
            hyps.append(hyp)
            refs.append(ref)

            # score = self.compute_bleu([ref], [hyp], smooth=True)[1]
            # total_score += score
            # count += 1
            # ind_score[id] = score["bleu-4"]

        # avg_score = total_score / count
        corpus_bleu, bleu_order = self.compute_bleu(refs, hyps, smooth=True)
        # return corpus_bleu, avg_score, ind_score
        return corpus_bleu, bleu_order

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

class Rouge(object):
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    '''
    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param gts: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = list(gts.keys())

        score = dict()
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            score[id] = self.calc_score(hypo, ref)

        average_score = np.mean(np.array(list(score.values())))
        return average_score, score

    def method(self):
        return "Rouge"


def enc(s):
    return s.encode('utf-8')
def dec(s):
    return s.decode('utf-8')
METEOR_JAR  = 'data/meteor-1.5.jar'
class Meteor(object):
    def __init__(self):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                v = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(dec(v.strip())))
                except:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))
                    # You can try uncommenting the next code line to show stderr from the Meteor JAR.
                    # If the Meteor JAR is not writing to stderr, then the line will just hang.
                    # sys.stderr.write("Error from Meteor:\n{}".format(self.meteor_p.stderr.read()))
                    raise
            score = float(dec(self.meteor_p.stdout.readline()).strip())

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats 
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        self.close()

if __name__ == "__main__":
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

        corpus_bleu, bleu_order, ind_score = Bleu().corpus_bleu(hypotheses=predictions_dict, references=references_dict)
        print("corpus_bleu : ", corpus_bleu)    # corpus_bleu : 0.25467977003051817
        print("Bleu order1-4 : ", bleu_order)   # {1: 0.45147210394009457, 2: 0.25286423679476816, 3: 0.20369564445128535, 4: 0.18091631042057468}
        print("ind score len = {}".format(len(ind_score)))
        with open("transformer_out_inds_score",'w') as fw:
            for value in ind_score.values():
                fw.write(f"{value}\n")

        # FIXME Meteor has something error!
        score, _ = Meteor().compute_score(gts=references_dict, res=predictions_dict)
        print("meteor : ", score)   # meteor :  0.20701547506449144

        rouge_l_score, _ = Rouge().compute_score(gts=references_dict, res=predictions_dict)
        print("rouge-l : ", rouge_l_score)  # rouge-l :  0.47184490911050164
