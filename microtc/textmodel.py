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
from .utils import get_class, SparseMatrix
from typing import Union


PUNCTUACTION = ";:,.@\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~<>|"
SKIP_SYMBOLS = set(";:,.@\\-\"/" + SYMBOLS)
SKIP_SYMBOLS_AND_SPACES = set(";:,.@\\-\"/" + SYMBOLS + '\t\n\r ')
# SKIP_WORDS = set(["…", "..", "...", "...."])
WEIGHTING = dict(tfidf="microtc.weighting.TFIDF",
                 tf="microtc.weighting.TF",
                 entropy="microtc.weighting.Entropy")


def norm_chars(text, del_diac=True, del_dup=True, del_punc=False):
    """
    Transform text by removing diacritics, duplicates, and punctuation.
    It adds ~ at the beginning, the end, and the spaces are changed by ~.

    :param text: Text
    :type text: str
    :param del_diac: Delete diacritics
    :type del_diac: bool
    :param del_dup: Delete duplicates
    :type del_dup: bool
    :param del_punc: Delete punctuation symbols
    :type del_punc: bool
    :rtype: str

    Example:

    >>> from microtc.textmodel import norm_chars
    >>> norm_chars("Life is good at Méxicoo.")
    '~Life~is~god~at~Mexico.~'

    """
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


def get_word_list(text):
    """
    Transform a text (begining and ending with ~) to list words.
    It is called after :py:func:`microtc.textmodel.norm_chars`.

    Example

    >>> from microtc.textmodel import get_word_list
    >>> get_word_list("~Someone's house.~")
    ["Someone's", 'house']

    :param text: text
    :type text: str

    :rtype: list
    """

    L = []
    prev = ' '
    for u in text[1:len(text)-1]:
        if u in SKIP_SYMBOLS:
            u = ' '

        if prev == ' ' and u == ' ':
            continue

        if prev == ' ' and u == "'":
            continue

        L.append(u)
        prev = u

    return ("".join(L)).split()


def expand_qgrams(text, qsize, output):
    """Expands a text into a set of q-grams

    :param text: Text
    :type text: str
    :param qsize: q-gram size
    :type qsize: int
    :param output: output
    :type output: list

    :returns: output
    :rtype: list

    Example:

    >>> from microtc.textmodel import expand_qgrams
    >>> output = list()
    >>> expand_qgrams("Good morning.", 3, output)
    ['q:Goo', 'q:ood', 'q:od ', 'q:d m', 'q: mo', 'q:mor', 'q:orn', 'q:rni', 'q:nin', 'q:ing', 'q:ng.']
    """

    _ = ["".join(a) for a in zip(*[text[i:] for i in range(qsize)])]
    [output.append("q:" + x) for x in _]
    return output


def expand_qgrams_word_list(wlist, qsize, output, sep='~'):
    """Expands a list of words into a list of q-grams. It uses `sep` to join words

    :param wlist: List of words computed by :py:func:`microtc.textmodel.get_word_list`.
    :type wlist: list
    :param qsize: q-gram size of words
    :type qsize: int
    :param output: output
    :type output: list
    :param sep: String used to join the words
    :type sep: str

    :returns: output
    :rtype: list

    Example:

    >>> from microtc.textmodel import expand_qgrams_word_list
    >>> wlist = ["Good", "morning", "Mexico"]
    >>> expand_qgrams_word_list(wlist, 2, list())
    ['Good~morning', 'morning~Mexico']
    """

    n = len(wlist)

    for start in range(n - qsize + 1):
        t = sep.join(wlist[start:start+qsize])
        output.append(t)

    return output


def expand_skipgrams_word_list(wlist, qsize, output, sep='~'):
    """Expands a list of words into a list of skipgrams. It uses `sep` to join words

    :param wlist: List of words computed by :py:func:`microtc.textmodel.get_word_list`.
    :type wlist: list
    :param qsize: (qsize, skip) qsize is the q-gram size and skip is the number of words ahead.
    :type qsize: tuple
    :param output: output
    :type output: list
    :param sep: String used to join the words
    :type sep: str

    :returns: output
    :rtype: list

    Example:

    >>> from microtc.textmodel import expand_skipgrams_word_list
    >>> wlist = ["Good", "morning", "Mexico"]
    >>> expand_skipgrams_word_list(wlist, (2, 1), list())
    ['Good~Mexico']

    """
    n = len(wlist)
    qsize, skip = qsize
    for start in range(n - (qsize + (qsize - 1) * skip) + 1):
        if qsize == 2:
            t = wlist[start] + sep + wlist[start+1+skip]
        else:
            t = sep.join([wlist[start + i * (1+skip)] for i in range(qsize)])

        output.append(t)

    return output


class TextModel(SparseMatrix):
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
    :param q_grams_words: Compute q-grams only on words
    :type q_grams_words: bool

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

    >>> vector = textmodel['categorizacion ingoetec']
    >>> vector2 = textmodel['cat']

    Using a different token_list

    >>> textmodel = TextModel(token_list=[[2, 1], -1, 3, 4]).fit(corpus)
    >>> vector = textmodel['categorizacion ingoetec']
    >>> vector2 = textmodel['cat']

    Train a classifier

    >>> from sklearn.svm import LinearSVC
    >>> y = [1, 0, 0]
    >>> textmodel = TextModel().fit(corpus)
    >>> m = LinearSVC().fit(textmodel.transform(corpus), y)
    >>> m.predict(textmodel.transform(corpus))
    array([1, 0, 0])
    """

    def __init__(self, docs=None, text: str='text',
                 num_option: str=OPTION_GROUP,
                 usr_option: str=OPTION_GROUP,
                 url_option: str=OPTION_GROUP,
                 emo_option: str=OPTION_GROUP,
                 hashtag_option: str=OPTION_NONE,
                 ent_option: str=OPTION_NONE,
                 lc: bool=True, del_dup: bool=True,
                 del_punc: bool=False, del_diac: bool=True,
                 token_list: list=[-1], 
                 token_min_filter: Union[int, float]=0,
                 token_max_filter: Union[int, float]=1,
                 select_ent: bool=False,
                 select_suff: bool=False, select_conn: bool=False,
                 weighting: str='tfidf',
                 q_grams_words: bool=False,
                 max_dimension: bool=False):
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
        self._q_grams_words = q_grams_words
        self._max_dimension = max_dimension
        if emo_option == OPTION_NONE:
            self.emo_map = None
        else:
            self.emo_map = EmoticonClassifier()

        if docs is not None and len(docs):
            self.fit(docs)

    @property
    def q_grams_words(self):
        try:
            return self._q_grams_words
        except AttributeError:
            return False

    @property
    def max_dimension(self):
        try:
            return self._max_dimension
        except AttributeError:
            return False

    # @property
    # def token_list(self):
    #     """Tokenizer parameters"""
    #     return self._token_list

    # @token_list.setter
    # def token_list(self, value):
    #     """
    #     >>> from microtc import TextModel
    #     >>> tm = TextModel()
    #     >>> tm.token_list = [-2, -1]
    #     >>> tm.token_list
    #     [-2, -1]
    #     """
    #     self._token_list = value
    #     for x in ['_q_grams', '_n_grams', '_skip_grams']:
    #         try:
    #             delattr(self, x)
    #         except AttributeError:
    #             continue

    @property
    def q_grams(self):
        """q-grams of characters
        >>> from microtc import TextModel
        >>> tm = TextModel(token_list=[-1, 3, (2, 1)])
        >>> tm.q_grams
        [3]
        """
        try:
            q_grams = self._q_grams
        except AttributeError:
            q_grams = [x for x in self.token_list if isinstance(x, int) and x > 0]
            self._q_grams = q_grams
        return q_grams

    @property
    def n_grams(self):
        """n-grams of words
        >>> from microtc import TextModel
        >>> tm = TextModel(token_list=[-1, 3, (2, 1)])
        >>> tm.n_grams
        [-1]
        """
        try:
            output = self._n_grams
        except AttributeError:
            output = [x for x in self.token_list if isinstance(x, int) and x < 0]
            self._n_grams = output
        return output

    @property
    def skip_grams(self):
        """skip-grams
        >>> from microtc import TextModel
        >>> tm = TextModel(token_list=[-1, 3, (2, 1)])
        >>> tm.skip_grams
        [(2, 1)]
        """
        try:
            output = self._skip_grams
        except AttributeError:
            output = [x for x in self.token_list if not isinstance(x, int)]
            self._skip_grams = output
        return output           

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
                                               token_max_filter=self.token_max_filter,
                                               max_dimension=self.max_dimension)
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
        odict_keys(['docs', 'text', 'num_option', 'usr_option', 'url_option', 'emo_option', 'hashtag_option', 'ent_option', 'lc', 'del_dup', 'del_punc', 'del_diac', 'token_list', 'token_min_filter', 'token_max_filter', 'select_ent', 'select_suff', 'select_conn', 'weighting', 'q_grams_words', 'max_dimension'])
        """

        import inspect
        sig = inspect.signature(cls)
        params = sig.parameters.keys()
        return params

    def transform(self, texts):
        """Convert test into a vector

        :param texts: List of text to be transformed
        :type texts: list

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias catedras', 'catedras conacyt']
        >>> textmodel = TextModel().fit(corpus)
        >>> X = textmodel.transform(corpus)
        """
        return self.tonp([self.__getitem__(x) for x in texts])

    def vectorize(self, text):
        raise RuntimeError('Not implemented')

    def tokenize(self, text):
        """Transform text to tokens.
        The procedure is:

        - :py:func:`microtc.textmodel.TextModel.text_transformations`.
        - :py:func:`microtc.textmodel.TextModel.compute_tokens`.
        - :py:func:`microtc.textmodel.TextModel.select_tokens`.

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

        if isinstance(text, dict):
            text = self.get_text(text)

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

    @property
    def disable_text_transformations(self):
        try:
            return self._disable_text_transformations
        except AttributeError:
            return False

    @disable_text_transformations.setter
    def disable_text_transformations(self, v):
        self._disable_text_transformations = v

    def text_transformations(self, text):
        """
        Text transformations. It starts by analyzing emojis, hashtags, entities,
        lower case, numbers, URL, and users. After these transformations are applied
        to the text, it calls :py:func:`microtc.textmodel.norm_chars`.

        :param text:
        :type text: str

        :rtype: str

        Example:

        >>> from microtc.textmodel import TextModel
        >>> tm = TextModel(del_dup=False)
        >>> tm.text_transformations("Life is good at México @mgraffg.")
        '~life~is~good~at~mexico~_usr~'
        """

        if text is None:
            text = ''

        if isinstance(text, dict):
            text = self.get_text(text)

        if self.disable_text_transformations:
            return text

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
            text = re.sub(r"\d\d*\.?\d*|\d*\.\d\d*", "", text)
        elif self.num_option == OPTION_GROUP:
            text = re.sub(r"\d\d*\.?\d*|\d*\.\d\d*", "_num", text)

        if self.url_option == OPTION_DELETE:
            text = re.sub(r"https?://\S+", "", text)
        elif self.url_option == OPTION_GROUP:
            text = re.sub(r"https?://\S+", "_url", text)

        if self.usr_option == OPTION_DELETE:
            text = re.sub(r"@\S+", "", text)
        elif self.usr_option == OPTION_GROUP:
            text = re.sub(r"@\S+", "_usr", text)

        return norm_chars(text, del_diac=self.del_diac, del_dup=self.del_dup,
                          del_punc=self.del_punc)

    def get_word_list(self, *args, **kwargs):
        return get_word_list(*args, **kwargs)

    def compute_n_grams(self, textlist):
        output = []
        for q in self.n_grams:
            expand_qgrams_word_list(textlist, abs(q), output)
        return output

    def compute_skip_grams(self, textlist):
        output = []
        for q in self.skip_grams:
            expand_skipgrams_word_list(textlist, q, output)
        return output

    def compute_q_grams(self, text):
        output = []
        for q in self.q_grams:
            expand_qgrams(text, q, output)
        return output

    def compute_q_grams_words(self, textlist):
        """
        >>> from microtc import TextModel
        >>> tm = TextModel(token_list=[3])
        >>> tm.compute_q_grams_words(['abc', 'def'])
        ['q:~ab', 'q:abc', 'q:bc~', 'q:~de', 'q:def', 'q:ef~']
        """
        output = []
        textlist = ['~' + x + '~' for x in textlist]
        for qsize in self.q_grams:
            _ = qsize - 1
            extra = [x for x in textlist if len(x) >= _]
            qgrams = [["".join(output) for output in zip(*[text[i:] for i in range(qsize)])] 
                      for text in extra]
            for _ in qgrams:
                [output.append("q:" + x) for x in _]
        return output       

    def compute_tokens(self, text):
        """
        Compute tokens from a text using q-grams of characters and words, and skip-grams.

        :param text: Text transformed by :py:func:`microtc.textmodel.TextModel.text_transformations`.
        :type text: str

        :rtype: list

        Example:

        >>> from microtc.textmodel import TextModel
        >>> tm = TextModel(token_list=[-2, -1])
        >>> tm.compute_tokens("~Good morning~")
        [['Good~morning', 'Good', 'morning'], [], []]
        >>> tm = TextModel(token_list=[3])
        >>> tm.compute_tokens('abc def')
        [[], [], ['q:abc', 'q:bc ', 'q:c d', 'q: de', 'q:def']]
        >>> tm = TextModel(token_list=[(2, 1)])
        >>> tm.compute_tokens('~abc x de~')
        [[], ['abc~de'], []]
        >>> tm = TextModel(token_list=[3], q_grams_words=True)
        >>> tm.compute_tokens('~abc def~')
        [[], [], ['q:~ab', 'q:abc', 'q:bc~', 'q:~de', 'q:def', 'q:ef~']]
        """
        L = []
        textlist = self.get_word_list(text)
        L.append(self.compute_n_grams(textlist))
        L.append(self.compute_skip_grams(textlist))
        if self.q_grams_words:
            L.append(self.compute_q_grams_words(textlist))
        else:
            L.append(self.compute_q_grams(text))
        return L

    def select_tokens(self, L):
        """
        Filter tokens using suffix or connections

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

        return self.model.num_terms

    @property
    def token_weight(self):
        """
        Weight associated to each token id

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
        >>> textmodel = TextModel().fit(corpus)
        >>> _ = textmodel.transform(corpus)
        >>> textmodel.token_weight[5]
        1.584962500721156
        """
        return self.model.wordWeight

    @property
    def id2token(self):
        """
        Token identifier to token

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
        >>> textmodel = TextModel().fit(corpus)
        >>> _ = textmodel.transform(corpus)
        >>> textmodel.id2token[5]
        'de'
        """
        try:
            return self._id2token
        except AttributeError:
            self._id2token = {v:k for k, v in self.token2id.items()}
            return self._id2token

    @property
    def token2id(self):
        """
        Token to token identifier

        >>> from microtc.textmodel import TextModel
        >>> corpus = ['buenos dias', 'catedras conacyt', 'categorizacion de texto ingeotec']
        >>> textmodel = TextModel().fit(corpus)
        >>> _ = textmodel.transform(corpus)
        >>> textmodel.token2id['de']
        5
        """
        return self.model.word2id


