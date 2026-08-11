#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROUGE-L scorer ported from R2GenGPT/evalcap/rouge (Lin & Hovy, 2004).

Self-contained: no external jar / java dependencies.
"""
import numpy as np


def my_lcs(string, sub):
    """Longest common subsequence length between two tokenized strings."""
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0] * (len(sub) + 1) for _ in range(len(string) + 1)]
    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    return lengths[len(string)][len(sub)]


class Rouge(object):
    """ROUGE-L wrapper."""

    def __init__(self, beta=1.2):
        self.beta = beta

    def calc_score(self, candidate, refs):
        prec = []
        rec = []
        token_c = candidate[0].split(" ")
        for reference in refs:
            token_r = reference.split(" ")
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / \
                    float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys(), "reference/hypothesis id sets differ"
        imgIds = list(gts.keys())
        score = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0
            score.append(self.calc_score(hypo, ref))
        average_score = np.mean(np.array(score))
        return average_score, np.array(score)

    def method(self):
        return "Rouge"
