#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Readers for the pke module."""

import xml.etree.ElementTree as etree
import spacy
import json

from pke.data_structures import Document


class Reader(object):
    def read(self, path):
        raise NotImplementedError


class MinimalCoreNLPReader(Reader):
    """Minimal CoreNLP XML Parser."""

    def __init__(self):
        self.parser = etree.XMLParser()

    def read(self, path, **kwargs):
        sentences = []
        tree = etree.parse(path, self.parser)
        for sentence in tree.iterfind('./document/sentences/sentence'):
            # get the character offsets
            starts = [int(u.text) for u in
                      sentence.iterfind("tokens/token/CharacterOffsetBegin")]
            ends = [int(u.text) for u in
                    sentence.iterfind("tokens/token/CharacterOffsetEnd")]
            sentences.append({
                "words": [u.text for u in
                          sentence.iterfind("tokens/token/word")],
                "lemmas": [u.text for u in
                           sentence.iterfind("tokens/token/lemma")],
                "POS": [u.text for u in sentence.iterfind("tokens/token/POS")],
                "char_offsets": [(starts[k], ends[k]) for k in
                                 range(len(starts))]
            })
            sentences[-1].update(sentence.attrib)

        doc = Document.from_sentences(sentences, input_file=path, **kwargs)

        return doc

class JsonTextReader(Reader):
    """Spacy JSON  Parser."""

    def read(self, text, **kwargs):
        obj = json.loads(text)

        sentences = []
        for sentence_id, s in enumerate(obj['sents']):
            sentences.append({
                "words": [u['t'] for u in s['tok']],
                "lemmas": [u.get('l', '') for u in s['tok']],
                "POS": [u['p'] for u in s['tok']],
                "char_offsets": [(u['o'], u['o'] + len(u['t'])) for u in s['tok']]
            })

        doc = Document.from_sentences(sentences,
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)

        return doc

class RawTextReader(Reader):
    """Reader for raw text."""

    nlps = {}

    @staticmethod
    def set_nlp(nlp, language):
        RawTextReader.nlps[language] = nlp

    def __init__(self, language=None):
        """Constructor for RawTextReader.

        Args:
            language (str): language of text to process.
        """

        self.language = language

        if language is None:
            self.language = 'en'

    def read(self, text, **kwargs):
        """Read the input file and use spacy to pre-process.

        Args:
            text (str): raw text to pre-process.
            max_length (int): maximum number of characters in a single text for
                spacy, default to 1,000,000 characters (1mb).
        """

        max_length = kwargs.get('max_length', 10**6)

        if self.language in RawTextReader.nlps:
            nlp = RawTextReader.nlps[self.language]
        else:
            nlp = spacy.load(self.language,
                            max_length=max_length)

        spacy_doc = nlp(text)

        sentences = []
        for sentence_id, sentence in enumerate(spacy_doc.sents):
            sentences.append({
                "words": [token.text for token in sentence],
                "lemmas": [token.lemma_ for token in sentence],
                "POS": [token.pos_ for token in sentence],
                "char_offsets": [(token.idx, token.idx + len(token.text))
                                     for token in sentence]
            })

        doc = Document.from_sentences(sentences,
                                      input_file=kwargs.get('input_file', None),
                                      **kwargs)

        return doc

