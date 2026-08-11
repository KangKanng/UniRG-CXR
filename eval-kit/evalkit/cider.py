#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CIDEr scorer ported from R2GenGPT/evalcap/cider (Vedantam, Zitnick, Parikh).

Self-contained: no external jar / java dependencies.
"""
import math
from collections import defaultdict

import numpy as np
def precook(s, n=4, out=False):
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    return precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer."""

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def copy(self):
        new = CiderScorer(n=self.n, sigma=self.sigma)
        new.ctest = self.ctest[:]
        new.crefs = [r[:] for r in self.crefs]
        return new

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                self.ctest.append(cook_test(test, self.n))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), \
            "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        for refs in self.crefs:
            for ngram in set([ng for ref in refs for (ng, c) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                for (ngram, count) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        self.ref_len = np.log(float(len(self.crefs)))
        if len(self.crefs) == 1:
            self.ref_len = 1
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score)
            score_avg /= len(refs)
            score_avg *= 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        self.compute_doc_freq()
        assert len(self.ctest) >= max(self.document_frequency.values()), \
            "document frequency larger than # of test sentences"
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


class Cider(object):
    """CIDEr wrapper exposing compute_score(gts, res)."""

    def __init__(self, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys(), "reference/hypothesis id sets differ"
        imgIds = list(gts.keys())
        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            cider_scorer += (hypo[0], ref)
        (score, scores) = cider_scorer.compute_score()
        return score, scores

    def method(self):
        return "CIDEr"
