# Copyright 2018 Mario Graff

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from collections import Counter
import os
from typing import Union


TEXT = os.environ.get("TEXT", 'text')
KLASS = os.environ.get("KLASS", 'klass')
VALUE = os.environ.get("VALUE", 'value')


class TFIDF(object):
    """
    Vector Space model using TFIDF

    :param docs: corpus as a list of list of tokens
    :type docs: list
    :param X: original corpus, useful to pass extra information in a dict
    :type X: list
    :param token_min_filter: Keep those tokens that appear more times than the parameter
    :type token_min_filter: int or float

    :param token_max_filter: Keep those tokens that appear less times than the parameter
    :type token_max_filter: int or float

    Usage:

    >>> from microtc.weighting import TFIDF
    >>> tokens = [['buenos', 'dia', 'microtc'], ['excelente', 'dia'], ['buenas', 'tardes'], ['las', 'vacas', 'me', 'deprimen'], ['odio', 'los', 'lunes'], ['odio', 'el', 'trafico'], ['la', 'computadora'], ['la', 'mesa'], ['la', 'ventana']]
    >>> tfidf = TFIDF(tokens)
    >>> vector = tfidf['buenos', 'X', 'trafico']
    """

    def __init__(self, docs, X=None,
                 token_min_filter: Union[int, float]=0,
                 token_max_filter: Union[int, float]=1,
                 max_dimension: bool=False,
                 unit_vector: bool=True):
        self.unit_vector = unit_vector
        w2id = {}
        weight = {}
        self._ndocs = len(docs)
        for tokens in docs:
            for x, freq in Counter(tokens).items():
                try:
                    ident = w2id[x]
                    weight[ident] = weight[ident] + 1
                except KeyError:
                    ident = len(w2id)
                    w2id[x] = ident
                    weight[ident] = 1
        # remove tokens with freq = N
        w2id = [(k, v) for k, v in w2id.items() if weight[v] < self._ndocs]
        w2id.sort(key=lambda x: x[1])
        mm = {k: v[1] for k, v in enumerate(w2id)}
        w2id = {v[0]: k for k, v in enumerate(w2id)}
        weight = {ident: weight[mm[ident]] for ident in w2id.values()}

        if not max_dimension and (token_min_filter > 0 or token_max_filter != 1):
            if token_min_filter < 1:
                token_min_filter = int(self._ndocs * token_min_filter)
                if token_min_filter < 1:
                    token_min_filter = 1
            if token_min_filter > 0:
                w2id = [(k, v) for k, v in w2id.items() if weight[v] > token_min_filter]
            if token_max_filter != 1:
                if token_max_filter < 1:
                    token_max_filter = int(self._ndocs * token_max_filter)
                w2id = [(k, v) for k, v in w2id if weight[v] < token_max_filter]
            w2id.sort(key=lambda x: x[1])
            mm = {k: v[1] for k, v in enumerate(w2id)}
            w2id = {v[0]: k for k, v in enumerate(w2id)}
            weight = {ident: weight[mm[ident]] for ident in w2id.values()}
        elif max_dimension:
            assert isinstance(token_max_filter, int) and token_max_filter > 1
            id2word = {v: k for k, v in w2id.items()}
            word_weight = [[v , id2word[k]] for k, v in weight.items()]
            word_weight.sort(key=lambda x: x[0], reverse=True)
            word_weight = word_weight[:token_max_filter]
            w2id = {token: k for k, (w, token) in enumerate(word_weight)}
            weight = {k: w for k, (w, token) in enumerate(word_weight)}
        self.word2id = w2id
        self.wordWeight = weight

    @property
    def unit_vector(self):
        try:
            return self._unit_vector
        except AttributeError:
            self._unit_vector = True
        return self._unit_vector

    @unit_vector.setter
    def unit_vector(self, value):
        self._unit_vector = value

    @property
    def num_terms(self):
        """Number of terms"""

        return self._num_terms

    @property
    def word2id(self):
        """Map word to id"""

        return self._w2id

    @word2id.setter
    def word2id(self, value):
        self._num_terms = len(value)
        self._w2id = value

    @property
    def wordWeight(self):
        """Weight associated to each word, this could be the inverse document frequency"""
        return self._weight

    @wordWeight.setter
    def wordWeight(self, value):
        """Inverse document frequency

        :param value: weights
        :type value: dict
        """

        N = self._ndocs
        self._weight = {k: np.log2(N / v) for k, v in value.items()}

    def doc2weight(self, tokens):
        """Weight associated to each token

        :param tokens: list of tokens
        :type tokens: lst

        :rtype: tuple - ids, term frequency, wordWeight
        """
        lst = []
        w2id = self._w2id
        weight = self.wordWeight
        for token in tokens:
            try:
                id = w2id[token]
                lst.append(id)
            except KeyError:
                continue
        ids_tf = [(a, b) for a, b in Counter(lst).items()]
        # ids, tf = np.unique(lst, return_counts=True)
        ids = [x[0] for x in ids_tf]
        tf = np.array([x[1] for x in ids_tf])
        tf = tf / tf.sum()
        df = np.array([weight[x] for x in ids])
        return ids, tf, df

    def __getitem__(self, tokens):
        """
        TF-IDF and the vectors are normalised.

        :param tokens: list of tokens
        :type tokens: lst

        :rtype: lst
        """

        __ = self.doc2weight(tokens)
        r = [(i, _tf * _df) for i, _tf, _df in zip(*__)]
        if not self.unit_vector:
            return r
        n = np.sqrt(sum([x * x for _, x in r]))
        return [(i, x/n) for i, x in r]

    @staticmethod
    def filter(counter, token_min_filter=0.001, token_max_filter=0.999):
        N = counter.update_calls
        if token_min_filter > 0 or token_max_filter != 1:
            if token_min_filter < 1:
                token_min_filter = int(N * token_min_filter)
                if token_min_filter < 1:
                    token_min_filter = 1
            if token_min_filter > 0:
                keys = [k for k, v in counter.items() if v <= token_min_filter]
                for k in keys:
                    del counter[k]
            if token_max_filter != 1:
                if token_max_filter < 1:
                    token_max_filter = int(N * token_max_filter)
                keys = [k for k, v in counter.items() if v >= token_max_filter]
                for k in keys:
                    del counter[k]
        return counter

    @classmethod
    def counter(cls, counter, token_min_filter=0, token_max_filter=1):
        """
        Create from :py:class:`microtc.utils.Corpus`

        :param counter: Tokens
        :param type: :py:class:`microtc.utils.Corpus`
        """

        cls.filter(counter, token_min_filter=token_min_filter,
                    token_max_filter=token_max_filter)
        ins = cls([])
        N = counter.update_calls
        ins._ndocs = N
        words = list(counter.keys())
        words.sort()
        word2id = {w: i for i, w in enumerate(words)}
        weight = {word2id[k]: np.log2(N) - np.log2(v) for k, v in counter.items()}
        ins._weight = weight
        ins.word2id = word2id
        return ins


class TF(TFIDF):
    @property
    def wordWeight(self):
        """Weight associated to each word, this is one on TF"""
        return self._weight

    @wordWeight.setter
    def wordWeight(self, value):
        """Inverse document frequency

        :param value: weights
        :type value: dict
        """

        self._weight = {k: 1 for k, v in value.items()}

    def __getitem__(self, tokens):
        """
        TF, the frequency is normalised

        :param tokens: list of tokens
        :type tokens: lst

        :rtype: lst
        """

        __ = self.doc2weight(tokens)
        r = [(i, _tf) for i, _tf, _df in zip(*__)]
        return r


class Entropy(TF):
    """
    Vector Space using 1 - entropy as the weighting scheme

    Usage:

    >>> from microtc.weighting import Entropy
    >>> tokens = [['buenos', 'dia', 'microtc'], ['excelente', 'dia'], ['buenas', 'tardes'], ['las', 'vacas', 'me', 'deprimen', 'al', 'dia'], ['odio', 'los', 'lunes'], ['odio', 'el', 'trafico'], ['la', 'computadora'], ['la', 'mesa'], ['la', 'ventana']]
    >>> y = [0, 0, 0, 2, 2, 2, 1, 1, 1]
    >>> ent = Entropy(tokens, X=[dict(text=t, klass=k) for t, k in zip(tokens, y)])
    >>> vector = ent['buenos', 'X', 'dia']
    """
    def __init__(self, docs, X=None, **kwargs):
        assert X is not None
        super(Entropy, self).__init__(docs, X=X, **kwargs)
        self.wordWeight = self.entropy(docs, X, self.word2id)

    @property
    def wordWeight(self):
        """Weight associated to each word, entropy per token"""
        return self._weight

    @wordWeight.setter
    def wordWeight(self, value):
        """Entropy

        :param value: weights
        :type value: dict
        """

        if isinstance(value, dict):
            self._weight = value
        else:
            self._weight = {k: v for k, v in enumerate(value)}

    @staticmethod
    def entropy(corpus, docs, word2id):
        """
        Compute entropy

        :param corpus: Tokenized corpus, i.e., as a list of tokens list
        :type corpus: list
        :param docs: Original corpus is a list of dictionaries where key klass contains the class or label
        :type docs: list
        :param word2id: Map token to identifier
        :type word2id: dict

        :rtype: np.array
        """
        m = word2id
        y = [x[KLASS] for x in docs]
        klasses = np.unique(y)
        nklasses = klasses.shape[0]
        ntokens = len(m)
        # hist = np.ones((klasses.shape[0], ntokens))
        hist = np.full((klasses.shape[0], ntokens), 3)

        for ki, klass in enumerate(klasses):
            for _y, tokens in zip(y, corpus):
                if _y != klass:
                    continue
                for x in Counter(tokens).keys():
                    try:
                        hist[ki, m[x]] += 1
                    except KeyError:
                        continue

        # hist = np.log2(hist + 1)
        hist = hist / hist.sum(axis=0)
        # hist[~np.isfinite(hist)] = 1.0 / nklasses
        logc = np.log2(hist)
        logc[~np.isfinite(logc)] = 0
        if nklasses > 2:
            logc = logc / np.log2(nklasses)
        return (1 + (hist * logc).sum(axis=0))

    def __getitem__(self, tokens):
        """
        Entropy

        :param tokens: list of tokens
        :type tokens: lst

        :rtype: lst
        """

        __ = self.doc2weight(tokens)
        r = [(i, _df) for i, _tf, _df in zip(*__)]
        return r
