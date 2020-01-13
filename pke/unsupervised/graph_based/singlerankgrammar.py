# -*- coding: utf-8 -*-
# Author: Dominique Sandoz
# Date: 13-01-2020

"""SingleRank keyphrase extraction model with grammar rules.

Simple extension of the SingleRank model (see singlerank.py), that
forces candidates to be weighted according to grammar rules as stated in the paper.

The idea is also found in the implementation of Single Topical PageRank (single_tpr.py),
where the grammar "NP:{<ADJ>*<NOUN|PROPN>+}" (UD tagset) is used.

@inproceedings{Wan:2008:SDK:1620163.1620205,
	Author = {Wan, Xiaojun and Xiao, Jianguo},
	Booktitle = {Proceedings of the 23rd National Conference on Artificial Intelligence - Volume 2},
	Pages = {855--860},
	Publisher = {AAAI Press},
	Series = {AAAI'08},
	Title = {Single Document Keyphrase Extraction Using Neighborhood Knowledge},
	Year = {2008}}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

from pke.unsupervised.graph_based.singlerank import SingleRank


class SingleRankGrammar(SingleRank):
    """SingleRank keyphrase extraction model.

    This model is an extension of the TextRank model that uses the number of
    co-occurrences to weigh edges in the graph.

    Parameterized example::

        import pke

        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.SingleRank()

        # 2. load the content of the document.
        extractor.load_document(input='path/to/input',
                                language='en',
                                normalization=None)

        # 3. select the noun phrases as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words of
        #    certain part-of-speech (nouns and adjectives) that are connected if
        #    they occur in a window of 10 words.
        extractor.candidate_weighting(window=10,
                                      pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for SingleRankGrammar."""

        super(SingleRank, self).__init__()

    def candidate_selection(self, grammar=None, **kwargs):
        """Candidate selection heuristic.

        Following Wan & Xiao, 2008:
        " ... only the nouns and adjectives are added ..."
        " ... sequences of adjacent candidate words are collapsed into a multi-word phrase..."
        " ... phrases ending with an adjective are not allowed [...] only phrases ending with a noun are collected ..."

        Args:
            grammar (str): grammar defining POS patterns of NPs, defaults to 
                "NP: {<ADJ>*<NOUN|PROPN>+}".
        """

        if grammar is None:
            grammar = "NP:{<ADJ>*<NOUN|PROPN>+}"

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)