#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this example uses TopicRank
from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal

from pke.unsupervised import SingleRankGrammar
from pke.unsupervised import SingleRank
from pke.unsupervised import EmbedRank

from pke import RawTextReader

import spacy
import time

nlp = spacy.load('en', disable=["ner", "parser"])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

RawTextReader.set_nlp(nlp, 'en')

print("EmbedRank++:")

startup_time = time.time()

# create a EmbedRank extractor
extractor = EmbedRank()

embedding_distributor = EmbeddingDistributorLocal('../../torontobooks_unigrams.bin')

startup_time = time.time() - startup_time
per_document_time = time.time()

# the input language is set to English (used for the stoplist)
extractor.load_document(input='C-1.txt',
                        language="en")

# select the keyphrase candidates, for SingleRankGrammar sequences that fulfill the given grammar
extractor.candidate_selection()

# weight the candidates using a random walk. The threshold parameter sets the
# minimum similarity for clustering, and the method parameter defines the 
# linkage method
extractor.candidate_weighting(embedding_distributor)

per_document_time = time.time() - per_document_time

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)

print("Startup time: " + str(startup_time * 1000) + " ms, extraction: " + str(per_document_time * 1000) + "ms")

startup_time = time.time()
print("SingleRankGrammar:")
# create a SingleRankGrammar extractor
extractor = SingleRankGrammar()

startup_time = time.time() - startup_time
per_document_time = time.time()

# the input language is set to English (used for the stoplist)
extractor.load_document(input='C-1.txt',
                        language="en")

# select the keyphrase candidates, for SingleRankGrammar sequences that fulfill the given grammar
extractor.candidate_selection()

# weight the candidates using a random walk. The threshold parameter sets the
# minimum similarity for clustering, and the method parameter defines the 
# linkage method
extractor.candidate_weighting(window=10,
                              pos={'NOUN', 'PROPN', 'ADJ'})

per_document_time = time.time() - per_document_time

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)
    
print("Startup time: " + str(startup_time * 1000) + " ms, extraction: " + str(per_document_time * 1000) + "ms")

print("SingleRank:")
startup_time = time.time()
# create a SingleRank extractor
extractor = SingleRank()

startup_time = time.time() - startup_time
per_document_time = time.time()

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

per_document_time = time.time() - per_document_time

# print the n-highest (10) scored candidates
for (keyphrase, score) in extractor.get_n_best(n=10):
    print(keyphrase, score)

print("Startup time: " + str(startup_time * 1000) + " ms, extraction: " + str(per_document_time * 1000) + "ms")