#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this example uses TopicRank
from pke.unsupervised import SingleRankGrammar
from pke.unsupervised import SingleRank

print("SingleRankGrammar:")
# create a SingleRankGrammar extractor
extractor = SingleRankGrammar()

# the input language is set to English (used for the stoplist)
extractor.load_document(input='C-1.txt',
                        language="en")

# select the keyphrase candidates, for SingleRankGrammer sequences that fulfill the given grammar
extractor.candidate_selection()

# weight the candidates using a random walk. The threshold parameter sets the
# minimum similarity for clustering, and the method parameter defines the 
# linkage method
extractor.candidate_weighting(window=10,
                              pos={'NOUN', 'PROPN', 'ADJ'})

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)

print("SingleRank:")
# create a SingleRank extractor
extractor = SingleRank()

# the input language is set to English (used for the stoplist)
extractor.load_document(input='C-1.txt',
                        language="en")

# select the keyphrase candidates, for SingleRank the longest sequences of 
# nouns and adjectives
extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})

# weight the candidates using a random walk. The threshold parameter sets the
# minimum similarity for clustering, and the method parameter defines the 
# linkage method
extractor.candidate_weighting(window=10,
                              pos={'NOUN', 'PROPN', 'ADJ'})

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)
