#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import re
from tensorflow.contrib import learn
from sklearn.base import BaseEstimator, TransformerMixin


class TextToBagVec(BaseEstimator, TransformerMixin):

    @staticmethod
    def tokenizer(iterator):
        for value in iterator:
            yield list(value)

    def __init__(self, max_length=None, min_frequency=1):
        self.max_length = max_length
        self.min_frequency = min_frequency - 1
        self.vocab_processor = None
        self.vocabulary_ = None

    def fit(self, X):
        if self.max_length is None:
            self.max_length = max([len(x) for x in X])
        vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_length,
                                                                       min_frequency=self.min_frequency,
                                                                  tokenizer_fn=TextToBagVec.tokenizer)
        vocab_processor.fit(X)
        self.vocabulary_ = vocab_processor.vocabulary_
        self.vocab_processor = vocab_processor
        return self

    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.array(list(self.vocab_processor.transform(X)))

    def transform(self, X, y=None):
        return np.array(list(self.vocab_processor.transform(X)))

    def reverse(self, X):
        return list(self.vocab_processor.reverse(X))
