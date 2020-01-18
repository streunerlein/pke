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

nlp = spacy.load('fr_core_news_md', disable=['parser','ner'],exclude=['parser'])
nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)

RawTextReader.set_nlp(nlp, 'fr')

startup_time = time.time()
print("SingleRankGrammar:")
# create a SingleRankGrammar extractor
extractor = SingleRankGrammar()

startup_time = time.time() - startup_time
per_document_time = time.time()

# the input language is set to English (used for the stoplist)
extractor.load_document(input='LCE-1981-11-17-a-i0004.txt',
                        language="fr")

# select the keyphrase candidates, for SingleRankGrammar sequences that fulfill the given grammar
extractor.candidate_selection(grammar='NP:{<NOUN.*|PROPN.*|ADJ>*<NOUN.*|PROPN.*>+<ADJ>*}')
# NP: {<ADJ>*<NOUN|PROPN>+}

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
extractor.load_document(input='LCE-1981-11-17-a-i0004.txt',
                        language="fr")

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


print("SingleRank JSON input:")
startup_time = time.time()
# create a SingleRank extractor
extractor = SingleRank()

startup_time = time.time() - startup_time
per_document_time = time.time()

f = open('LCE-1981-11-17-a-i0004.json', 'r')
fjson = f.read()

# the input language is set to English (used for the stoplist)
extractor.load_document(input=fjson,
                        is_json=True,
                        language="fr")

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

f = open('LCE-1981-11-17-a-i0004.txt', 'r')
d = nlp(f.read())

import json
print(json.dumps({'sents':[{
  "lg": "fr",
  "tok": [
    {
      't': token.text,
      'p': token.pos_,
      'l': token.lemma_
    } for token in s
  ]
} for s in d.sents]}))
