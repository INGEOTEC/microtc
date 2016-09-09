# Copyright 2016 Eric S. Tellez

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
import os
import unicodedata
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE
from .emoticons import get_compiled_map, transform_del, transform_replace_by_klass, EmoticonClassifier
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

PUNCTUACTION = ";:,.@\\-\"'/"
SYMBOLS = "()[]¿?¡!{}~<>|"
SKIP_SYMBOLS = set(PUNCTUACTION + SYMBOLS)
# SKIP_WORDS = set(["…", "..", "...", "...."])


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


def norm_chars(text, del_diac=True, del_dup1=True, del_punc=False):
    L = ['~']

    prev = '~'
    for u in unicodedata.normalize('NFD', text):
        if del_diac:
            o = ord(u)
            if 0x300 <= o and o <= 0x036F:
                continue
            
        if u in ('\n', '\r', ' ', '\t'):
            u = '~'
        elif del_dup1 and prev == u:
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
    def __init__(
            self,
            docs,
            num_option=OPTION_GROUP,
            usr_option=OPTION_GROUP,
            url_option=OPTION_GROUP,
            emo_option=OPTION_GROUP,
            lc=True,
            del_dup1=True,
            del_punc=False,
            del_diac=True,
            token_list=[-1],
            **kwargs
    ):
        self.del_diac = del_diac
        self.num_option = num_option
        self.usr_option = usr_option
        self.url_option = url_option
        self.emo_option = emo_option
        self.lc = lc
        self.del_dup1 = del_dup1
        self.del_punc = del_punc
        self.token_list = token_list

        self.kwargs = {k: v for k, v in kwargs.items() if k[0] != '_'}

        if emo_option == OPTION_NONE:
            self.emo_map = None
        else:
            # self.emo_map = get_compiled_map(os.path.join(os.path.dirname(__file__), 'resources', 'emoticons.json'))
            self.emo_map = EmoticonClassifier()

        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        self.model = TfidfModel(corpus)

    def __str__(self):
        return "[TextModel {0}]".format(dict(
            num_option=self.num_option,
            usr_option=self.usr_option,
            url_option=self.url_option,
            emo_option=self.emo_option,
            lc=self.lc,
            del_dup1=self.del_dup1,
            del_punc=self.del_punc,
            del_diac=self.del_diac,
            token_list=self.token_list,
            kwargs=self.kwargs
        ))

    def __getitem__(self, text):
        return self.model[self.dictionary.doc2bow(self.tokenize(text))]

    def tokenize(self, text):
        # print("tokenizing", str(self), text)
        if text is None:
            text = ''

        # if self.emo_option == OPTION_DELETE:
        #     text = transform_del(text, self.emo_map)
        # elif self.emo_option == OPTION_GROUP:
        #     text = transform_replace_by_klass(text, self.emo_map)
        if self.emo_map:
            text = self.emo_map.replace(text, option=self.emo_option)

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

        text = norm_chars(text, del_diac=self.del_diac, del_dup1=self.del_dup1, del_punc=self.del_punc)

        L = []
        textlist = None

        for q in self.token_list:
            if isinstance(q, int):
                if q < 0:
                    if textlist is None:
                        textlist = get_word_list(text)

                    expand_qgrams_word_list(textlist, abs(q), L)
                else:
                    expand_qgrams(text, q, L)
            else:
                if textlist is None:
                    textlist = get_word_list(text)

                expand_skipgrams_word_list(textlist, q, L)

        return L
    
