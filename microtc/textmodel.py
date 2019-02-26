# Copyright 2016-2017 Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import unicodedata
import numpy as np
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE
from .emoticons import EmoticonClassifier
import os
from scipy.sparse import csr_matrix
from .utils import get_class


PUNCTUACTION = ";:,.@\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~<>|"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
SKIP_SYMBOLS_AND_SPACES = set(PUNCTUACTION + SYMBOLS + '\t\n\r ')
# SKIP_WORDS = set(["…", "..", "...", "...."])
WEIGHTING = dict(tfidf="microtc.weighting.TFIDF",
                 tf="microtc.weighting.TFIDF",
                 entropy="microtc.weighting.Entropy")


def get_word_list(text):
    L = []
    prev = ' '
    for u in text[1:len(text)-1]:
        if u in SKIP_SYMBOLS:
            u = ' '

        if prev == ' ' and u == ' ':
            continue

        L.append(u)
        prev = u

    return ("".join(L)).split()


def norm_chars(text, del_diac=True, del_dup=True, del_punc=False):
    L = ['~']

    prev = '~'
    for u in unicodedata.normalize('NFD', text):
        if del_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                continue
            
        if u in ('\n', '\r', ' ', '\t'):
            u = '~'
        elif del_dup and prev == u:
            continue
        elif del_punc and u in SKIP_SYMBOLS:
            prev = u
            continue

        prev = u
        L.append(u)

    L.append('~')

    return "".join(L)


def expand_qgrams(text, qsize, output):
    """Expands a text into a set of q-grams"""
    n = len(text)
    for start in range(n - qsize + 1):
        output.append(text[start:start+qsize])

    return output


def expand_qgrams_word_list(wlist, qsize, output, sep='~'):
    """Expands a list of words into a list of q-grams. It uses `sep` to join words"""
    n = len(wlist)
    
    for start in range(n - qsize + 1):
        t = sep.join(wlist[start:start+qsize])
        output.append(t)

    return output


def expand_skipgrams_word_list(wlist, qsize, output, sep='~'):
    """Expands a list of words into a list of skipgrams. It uses `sep` to join words"""
    n = len(wlist)
    qsize, skip = qsize
    for start in range(n - (qsize + (qsize - 1) * skip) + 1):
        if qsize == 2:
            t = wlist[start] + sep + wlist[start+1+skip]
        else:
            t = sep.join([wlist[start + i * (1+skip)] for i in range(qsize)])

        output.append(t)

    return output


class TextModel:
    """

    :param docs: Corpus
    :type docs: list
    :param text: In the case corpus is a dict then text is the key containing the text
    :type text: str
    :param num_option: Transformations on numbers (none | group | delete)
    :type num_option: str
    :param usr_option: Transformations on users (none | group | delete)
    :type usr_option: str
    :param url_option: Transformations on urls (none | group | delete)
    :type url_option: str
    :param emo_option: Transformations on emojis and emoticons (none | group | delete)
    :type emo_option: str
    :param hashtag_option: Transformations on hashtag (none | group | delete)
    :type hashtag_option: str
    :param ent_option: Transformations on entities (none | group | delete)
    :type ent_option: str

    :param lc: Lower case
    :type lc: bool
    :param del_dup: Remove duplicates e.g. hooola -> hola
    :type del_dup: bool
    :param del_punc: Remove punctuation symbols
    :type del_punc: True
    :param del_diac: Remove diacritics
    :type del_diac: bool
    :param token_list: Tokens > 0 qgrams < 0 word-grams
    :type token_list: list
    :param token_min_filter: Keep those tokens that appear more times than the parameter (used in weighting class)
    :type token_min_filter: int or float
    :param token_max_filter: Keep those tokens that appear less times than the parameter (used in weighting class)
    :type token_max_filter: int or float

    :param select_ent:
    :type select_ent: bool
    :param select_suff:
    :type select_suff: bool
    :param select_conn:
    :type select_conn: bool

    :param weighting: Weighting scheme (tfidf | tf | entropy)
    :type weighting: class or str

    Usage:

    >>> from microtc.textmodel import TextModel
    >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']

    Using default parameters

    >>> textmodel = TextModel().fit(corpus)

    Represent a text whose words are in the corpus and one that does not

    >>> print(textmodel['categorizacion ingoetec'], textmodel['cat'])
    [(4, 1.0)] []

    Using a different token_list

    >>> textmodel = TextModel(token_list=[[2, 1], -1, 3, 4]).fit(corpus)
    >>> print(textmodel['categorizacion ingoetec'], '--', textmodel['cat'])
    [(25, 0.06774376041676866), (27, 0.06774376041676866), (28, 0.06774376041676866), (50, 0.06774376041676866), (51, 0.06774376041676866), (54, 0.1835524837678424), (55, 0.1835524837678424), (56, 0.1835524837678424), (57, 0.1835524837678424), (59, 0.1835524837678424), (60, 0.1835524837678424), (65, 0.1835524837678424), (66, 0.1835524837678424), (67, 0.1835524837678424), (76, 0.1835524837678424), (77, 0.1835524837678424), (78, 0.1835524837678424), (81, 0.1835524837678424), (82, 0.1835524837678424), (83, 0.1835524837678424), (84, 0.1835524837678424), (89, 0.1835524837678424), (91, 0.1835524837678424), (92, 0.1835524837678424), (97, 0.1835524837678424), (98, 0.1835524837678424), (99, 0.1835524837678424), (100, 0.1835524837678424), (101, 0.1835524837678424), (102, 0.1835524837678424), (110, 0.1835524837678424), (111, 0.1835524837678424), (114, 0.1835524837678424), (115, 0.1835524837678424)] -- [(27, 0.816496580927726), (50, 0.408248290463863), (51, 0.408248290463863)]

    Train a classifier

    >>> from sklearn.svm import LinearSVC
    >>> y = [1, 0, 0]
    >>> textmodel = TextModel().fit(corpus)
    >>> m = LinearSVC().fit(textmodel.transform(corpus), y)
    >>> m.predict(textmodel.transform(corpus))
    array([1, 0, 0])
    """

    def __init__(self, docs=None, text='text', num_option=OPTION_GROUP,
                 usr_option=OPTION_GROUP, url_option=OPTION_GROUP,
                 emo_option=OPTION_GROUP, hashtag_option=OPTION_NONE,
                 ent_option=OPTION_NONE, lc=True, del_dup=True, del_punc=False, del_diac=True,
                 token_list=[-1], token_min_filter=0,
                 token_max_filter=1, select_ent=False, select_suff=False, select_conn=False,
                 weighting='tfidf'):
        self._text = os.getenv('TEXT', default=text)
        self.del_diac = del_diac
        self.num_option = num_option
        self.usr_option = usr_option
        self.url_option = url_option
        self.emo_option = emo_option
        self.ent_option = ent_option
        self.select_ent = select_ent
        self.select_suff = select_suff
        self.select_conn = select_conn

        self.hashtag_option = hashtag_option
        self.lc = lc
        self.del_dup = del_dup
        self.del_punc = del_punc
        self.token_list = token_list
        self.token_min_filter = token_min_filter
        self.token_max_filter = token_max_filter
        self.weighting = weighting
        self.weighting = WEIGHTING.get(weighting, weighting)

        if emo_option == OPTION_NONE:
            self.emo_map = None
        else:
            self.emo_map = EmoticonClassifier()

        if docs is not None and len(docs):
            self.fit(docs)

    def fit(self, X):
        """
        Train the model

        :param X: Corpus
        :type X: list
        :rtype: instance
        """

        tokens = [self.tokenize(d) for d in X]
        self.model = get_class(self.weighting)(tokens, X=X,
                                               token_min_filter=self.token_min_filter,
                                               token_max_filter=self.token_max_filter)
        return self

    def __getitem__(self, text):
        """Convert text into a vector

        :param text: Text to be transformed
        :type text: str

        :rtype: list
        """
        return self.model[self.tokenize(text)]

    @classmethod
    def params(cls):
        """
        Parameters

        >>> from microtc.textmodel import TextModel
        >>> TextModel.params()
        odict_keys(['docs', 'text', 'num_option', 'usr_option', 'url_option', 'emo_option', 'hashtag_option', 'ent_option', 'lc', 'del_dup', 'del_punc', 'del_diac', 'token_list', 'token_min_filter', 'token_max_filter', 'select_ent', 'select_suff', 'select_conn', 'weighting'])
        """

        import inspect
        sig = inspect.signature(cls)
        params = sig.parameters.keys()
        return params

    def transform(self, texts):
        """Convert test into a vector

        :param texts: List of text to be transformed
        :type text: list

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias catedras', 'catedras conacyt']
        >>> textmodel = TextModel().fit(corpus)
        >>> X = textmodel.transform(corpus)
        >>> X.toarray()
        array([[0.70710678, 0.        , 0.70710678, 0.        ],
               [0.        , 0.        , 0.        , 1.        ]])
        """
        return self.tonp([self.__getitem__(x) for x in texts])

    def vectorize(self, text):
        raise RuntimeError('Not implemented')

    def tokenize(self, text):
        """Transform text to tokens

        :param text: Text
        :type text: str or list

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> tm = TextModel()
        >>> tm.tokenize("buenos dias")
        ['buenos', 'dias']
        >>> tm.tokenize(["buenos", "dias", "tenga usted"])
        ['buenos', 'dias', 'tenga', 'usted']
        """

        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(self._tokenize(_text))

            return tokens
        else:
            return self._tokenize(text)

    def get_text(self, text):
        """Return self._text key from text

        :param text: Text
        :type text: dict
        """

        return text[self._text]

    def text_transformations(self, text):
        """
        Text transformations

        :param text:
        :type text: str
        """

        if text is None:
            text = ''

        if isinstance(text, dict):
            text = self.get_text(text)

        if self.emo_map:
            text = self.emo_map.replace(text, option=self.emo_option)

        if self.select_ent:
            text = " ".join(re.findall(r"(@\S+|#\S+|[A-Z]\S+)", text))

        if self.hashtag_option == OPTION_DELETE:
            text = re.sub(r"#\S+", "", text)
        elif self.hashtag_option == OPTION_GROUP:
            text = re.sub(r"#\S+", "_htag", text)

        if self.ent_option == OPTION_DELETE:
            text = re.sub(r"[A-Z][a-z]+", "", text)
        elif self.ent_option == OPTION_GROUP:
            text = re.sub(r"[A-Z][a-z]+", "_ent", text)

        if self.lc:
            text = text.lower()

        if self.num_option == OPTION_DELETE:
            text = re.sub(r"\d+\.?\d+", "", text)
        elif self.num_option == OPTION_GROUP:
            text = re.sub(r"\d+\.?\d+", "_num", text)

        if self.url_option == OPTION_DELETE:
            text = re.sub(r"https?://\S+", "", text)
        elif self.url_option == OPTION_GROUP:
            text = re.sub(r"https?://\S+", "_url", text)

        if self.usr_option == OPTION_DELETE:
            text = re.sub(r"@\S+", "", text)
        elif self.usr_option == OPTION_GROUP:
            text = re.sub(r"@\S+", "_usr", text)

        return norm_chars(text, del_diac=self.del_diac, del_dup=self.del_dup, del_punc=self.del_punc)

    def compute_tokens(self, text):
        """
        :param text:
        :type text: str

        :rtype: list
        """
        L = []
        textlist = None

        _text = text
        for q in self.token_list:
            if isinstance(q, int):
                if q < 0:
                    if textlist is None:
                        textlist = get_word_list(text)
                    _ = list()
                    expand_qgrams_word_list(textlist, abs(q), _)
                    L.append(_)
                else:
                    _ = list()
                    expand_qgrams(_text, q, _)
                    L.append(_)
            else:
                if textlist is None:
                    textlist = get_word_list(text)

                _ = list()
                expand_skipgrams_word_list(textlist, q, _)
                L.append(_)

        return L

    def select_tokens(self, L):
        """
        :param L: list of tokens
        :type L: list

        :rtype: list
        """

        if self.select_suff:
            L = [tok for tok in L if tok[-1] in SKIP_SYMBOLS_AND_SPACES]
            
        if self.select_conn:
            L = [tok for tok in L if '~' in tok and tok[0] != '~' and tok[-1] != '~']
        return L

    def _tokenize(self, text):
        text = self.text_transformations(text)
        L = []
        for _ in self.compute_tokens(text):
            L += _
        L = self.select_tokens(L)
        if len(L) == 0:
            L = ['~']

        return L

    @property
    def num_terms(self):
        """Dimension which is the number of terms of the corpus

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
        >>> textmodel = TextModel().fit(corpus)
        >>> _ = textmodel.transform(corpus)
        >>> textmodel.num_terms
        8

        :rtype: int
        """

        try:
            return self._num_terms
        except AttributeError:
            self._num_terms = None
        return None

    def tonp(self, X):
        """Sparse representation to sparce matrix

        :param X: Sparse representation of matrix
        :type X: list
        :rtype: csr_matrix

        Example:

        >>> from microtc.textmodel import TextModel
        >>> tm = TextModel()
        >>> matrix = [[(1, 0.5), (3, -0.2)], [(2, 0.3)], [(0, 1), (3, -1.2)]]
        >>> r = tm.tonp(matrix)
        >>> r.toarray()
        array([[ 0. ,  0.5,  0. , -0.2],
               [ 0. ,  0. ,  0.3,  0. ],
               [ 1. ,  0. ,  0. , -1.2]])

        """

        if not isinstance(X, list):
            return X
        data = []
        row = []
        col = []
        for r, x in enumerate(X):
            cc = [_[0] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            col += cc
            data += [_[1] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            _ = [r] * len(cc)
            row += _
        if self.num_terms is None:
            _ = csr_matrix((data, (row, col)))
            self._num_terms = _.shape[1]
            return _
        return csr_matrix((data, (row, col)), shape=(len(X), self.num_terms))
