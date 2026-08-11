#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BLEU scorer ported from R2GenGPT/evalcap/bleu (MS-COCO caption eval).

Self-contained: no external jar / java dependencies.
"""
import copy
import math
from collections import defaultdict


def precook(s, n=4, out=False):
    """Return (length, ngram_counts) for a sentence string.

    ``out`` is kept for API parity with the original implementation but is
    unused (the original only used it to toggle a debug print path).
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return len(words), counts


def cook_refs(refs, n=4):
    """Cook a list of reference sentences for one segment."""
    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    return (reflen, maxcounts)


def cook_test(test, crefs, n=4):
    """Cook a test sentence given cooked references."""
    reflen, refmaxcounts = crefs[0], crefs[1]
    testlen, counts = precook(test, n)

    result = {}
    result["reflen"] = reflen
    result["testlen"] = testlen
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
    result['correct'] = [0] * n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)
    return result

class BleuScorer(object):
    __slots__ = ("n", "crefs", "ctest", "_score", "_ratio", "_testlen", "_reflen", "special_reflen")

    def copy(self):
        new = BleuScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        new._score = None
        return new

    def __init__(self, test=None, refs=None, n=4, special_reflen=None):
        self.n = n
        self.crefs = []
        self.ctest = []
        self.cook_append(test, refs)
        self.special_reflen = special_reflen

    def cook_append(self, test, refs):
        if refs is not None:
            self.crefs.append(cook_refs(refs, self.n))
            if test is not None:
                cooked_test = cook_test(test, self.crefs[-1], self.n)
                self.ctest.append(cooked_test)
            else:
                self.ctest.append(None)
        self._score = None

    def __iadd__(self, other):
        if type(other) is tuple:
            self.cook_append(other[0], other[1])
        else:
            assert self.compatible(other), "incompatible BLEUs."
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
            self._score = None
        return self

    def compatible(self, other):
        return isinstance(other, BleuScorer) and self.n == other.n

    def _single_reflen(self, reflens, option=None, testlen=None):
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = min((abs(l - testlen), l) for l in reflens)[1]
        else:
            assert False, "unsupported reflen option %s" % option
        return reflen

    def compute_score(self, option=None, verbose=0):
        n = self.n
        small = 1e-9
        tiny = 1e-15
        bleu_list = [[] for _ in range(n)]

        if self._score is not None:
            return self._score, bleu_list

        if option is None:
            option = "average" if len(self.crefs) == 1 else "closest"

        self._testlen = 0
        self._reflen = 0
        totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}

        for comps in self.ctest:
            testlen = comps['testlen']
            self._testlen += testlen

            if self.special_reflen is None:
                reflen = self._single_reflen(comps['reflen'], option, testlen)
            else:
                reflen = self.special_reflen

            self._reflen += reflen

            for key in ['guess', 'correct']:
                for k in range(n):
                    totalcomps[key][k] += comps[key][k]

            bleu = 1.
            for k in range(n):
                bleu *= (float(comps['correct'][k]) + tiny) \
                        / (float(comps['guess'][k]) + small)
                bleu_list[k].append(bleu ** (1. / (k + 1)))
            ratio = (testlen + tiny) / (reflen + small)
            if ratio < 1:
                for k in range(n):
                    bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

            if verbose > 1:
                print(comps, reflen)

        totalcomps['reflen'] = self._reflen
        totalcomps['testlen'] = self._testlen

        bleus = []
        bleu = 1.
        for k in range(n):
            bleu *= float(totalcomps['correct'][k] + tiny) \
                    / (totalcomps['guess'][k] + small)
            bleus.append(bleu ** (1. / (k + 1)))
        ratio = (self._testlen + tiny) / (self._reflen + small)
        if ratio < 1:
            for k in range(n):
                bleus[k] *= math.exp(1 - 1 / ratio)

        if verbose > 0:
            print(totalcomps)
            print("ratio:", ratio)

        self._score = bleus
        return self._score, bleu_list


class Bleu(object):
    """BLEU wrapper exposing compute_score(gts, res)."""

    def __init__(self, n=4):
        self._n = n

    def compute_score(self, gts, res, verbose=0):
        assert gts.keys() == res.keys(), "reference/hypothesis id sets differ"
        imgIds = list(gts.keys())

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1
            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option='closest', verbose=verbose)
        return score, scores

    def method(self):
        return "Bleu"
