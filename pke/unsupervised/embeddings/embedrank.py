# -*- coding: utf-8 -*-
# Author: Dominique Sandoz
# Date: 13-01-2020

"""EmbedRank and EmbedRank++ using s2v keyphrase extraction model with grammar rules for extraction.

This is basically a wrapper around the author's implementation on https://github.com/swisscom/ai-research-keyphrase-extraction

@article{DBLP:journals/corr/abs-1801-04470,
	Author = {Kamil Bennani{-}Smires and Claudiu Musat and Martin Jaggi and Andreea Hossmann and Michael Baeriswyl},
	Journal = {CoRR},
	Title = {EmbedRank: Unsupervised Keyphrase Extraction using Sentence Embeddings},
	Volume = {abs/1801.04470},
	Year = {2018}}
"""

import warnings

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging

from swisscom_ai.research_keyphrase.embeddings.emb_distrib_local import EmbeddingDistributorLocal
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.method import MMRPhrase, _MMR

from pke.base import LoadFile


class EmbedRank(LoadFile):
    """EmbedRank keyphrase extraction model.

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

        super(EmbedRank, self).__init__()

    def candidate_selection(self, grammar="NP:{<ADJ>*<NOUN|PROPN>+}", **kwargs):
        """Candidate selection heuristic.

        Following Wan & Xiao, 2008:
        " ... only the nouns and adjectives are added ..."
        " ... sequences of adjacent candidate words are collapsed into a multi-word phrase..."
        " ... phrases ending with an adjective are not allowed [...] only phrases ending with a noun are collected ..."

        Args:
            grammar (str): grammar defining POS patterns of NPs, defaults to 
                "NP: {<ADJ>*<NOUN|PROPN>+}".
        """

        """
        InputTextObj:
        Text tagget as a list of sentences
        where each sentence is a list of tuple (word, TAG).
        """

        tagged = [[(word, sentence.pos[i]) for i, word in enumerate(sentence.words)] for sentence in self.sentences]

        self.text_obj = InputTextObjTaggable(tagged, self.language)

        # select sequence of adjectives and nouns
        self.grammar_selection(grammar=grammar)

    def candidate_weighting(self, embedding_distributor, N=None, beta=0.65, alias_threshold=0.8):
        """Tailored candidate ranking method for TextRank. Keyphrase candidates
        are either composed from the T-percent highest-ranked words as in the
        original paper or extracted using the `candidate_selection()` method.
        Candidates are ranked using the sum of their (normalized?) words.

        Args:
            window (int): the window for connecting two words in the graph,
                defaults to 2.
            pos (set): the set of valid pos for words to be considered as nodes
                in the graph, defaults to ('NOUN', 'PROPN', 'ADJ').
            top_percent (float): percentage of top vertices to keep for phrase
                generation.
            normalized (False): normalize keyphrase score by their length,
                defaults to False.
        """

        if N is None:
            N = len(self.candidates)

        (candidates, relevances, aliases) = self.__MMRPhrase(embedding_distributor, self.text_obj, beta, N, alias_threshold=alias_threshold)

        forms_keys = [(' '.join(self.candidates[u].surface_forms[0]).lower(), u) for u in self.candidates.keys()]
        forms = dict(zip([f[0] for f in forms_keys], [f[1] for f in forms_keys]))
        
        self.aliases = {}

        for i, form in enumerate(candidates):
            key = forms[form]
            if (key in self.candidates):
              self.weights[key] = relevances[i]
              self.aliases[form] = aliases[i]
            else:
              logging.warning("Selected candidate not in candidates")

    def __extract_candidates_embedding_for_doc(self, embedding_distrib, inp_rpr):
      """
      from https://github.com/swisscom/ai-research-keyphrase-extraction but adapted to work with in-class candidates

      Return the list of candidate phrases as well as the associated numpy array that contains their embeddings.
      Note that candidates phrases extracted by PosTag rules  which are uknown (in term of embeddings)
      will be removed from the candidates.

      :param embedding_distrib: embedding distributor see @EmbeddingDistributor
      :param inp_rpr: input text representation see @InputTextObj
      :return: A tuple of two element containing 1) the list of candidate phrases
      2) a numpy array of shape (number of candidate phrases, dimension of embeddings :
      each row is the embedding of one candidate phrase
      """
      candidates = np.array([' '.join(self.candidates[u].surface_forms[0]).lower() for u in self.candidates.keys()])
      if len(candidates) > 0:
          embeddings = np.array(embedding_distrib.get_tokenized_sents_embeddings(candidates))  # Associated embeddings
          valid_candidates_mask = ~np.all(embeddings == 0, axis=1)  # Only candidates which are not unknown.
          return candidates[valid_candidates_mask], embeddings[valid_candidates_mask, :]
      else:
          return np.array([]), np.array([])

    def __MMRPhrase(self, embdistrib, text_obj, beta, N, use_filtered=True, alias_threshold=0.8):
      """
      from https://github.com/swisscom/ai-research-keyphrase-extraction but adapted to work with in-class candidates
      
      Extract N keyphrases

      :param embdistrib: embedding distributor see @EmbeddingDistributor
      :param text_obj: Input text representation see @InputTextObj
      :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
      :param N: number of keyphrases to extract
      :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
      :return: A tuple with 3 elements :
      1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
      2)list of associated relevance scores (list of float)
      3)list containing for each keyphrase a list of alias (list of list of string)
      """
      candidates, X = self.__extract_candidates_embedding_for_doc(embdistrib, text_obj)

      if len(candidates) == 0:
          warnings.warn('No keyphrase extracted for this document')
          return None, None, None

      return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold)

class InputTextObjTaggable(InputTextObj):
    """ Represent the input text in which we want to extract keyphrases

        Replacement fro InputTextObj from https://github.com/swisscom/ai-research-keyphrase-extraction
        to be able to customize considered_tags
    """

    def __init__(self, pos_tagged, lang, considered_tags = {'ADJ', 'NOUN', 'PROPN'}, stem=False, min_word_len=3):
        """
        :param pos_tagged: List of list : Text pos_tagged as a list of sentences
        where each sentence is a list of tuple (word, TAG).
        :param stem: If we want to apply stemming on the text.
        """
        self.min_word_len = min_word_len
        self.pos_tagged = []
        self.filtered_pos_tagged = []
        self.isStemmed = stem
        self.lang = lang

        self.considered_tags = considered_tags

        if stem:
            stemmer = PorterStemmer()
            self.pos_tagged = [[(stemmer.stem(t[0]), t[1]) for t in sent] for sent in pos_tagged]
        else:
            self.pos_tagged = [[(t[0].lower(), t[1]) for t in sent] for sent in pos_tagged]

        temp = []
        for sent in self.pos_tagged:
            s = []
            for elem in sent:
                if len(elem[0]) < min_word_len:
                    s.append((elem[0], 'LESS'))
                else:
                    s.append(elem)
            temp.append(s)

        self.pos_tagged = temp
        self.filtered_pos_tagged = [[(t[0].lower(), t[1]) for t in sent if self.is_candidate(t)] for sent in
                                    self.pos_tagged]