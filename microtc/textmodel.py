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
import unicodedata
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE
# from .emoticons import get_compiled_map, transform_del, transform_replace_by_klass, EmoticonClassifier
from .emoticons import EmoticonClassifier
# from .lang_dependency import LangDependency
import logging
import os

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
    def __init__(
            self,
            docs,
            num_option=OPTION_GROUP,
            usr_option=OPTION_GROUP,
            url_option=OPTION_GROUP,
            emo_option=OPTION_GROUP,
            hashtag_option=OPTION_NONE,
            lc=True,
            del_dup=True,
            del_punc=False,
            del_diac=True,
            get_conclusion=False,
            token_list=[-1],
            token_min_filter=-1,
            token_max_filter=1.0,
            tfidf=True,
            lang='arabic',
            neg=True,
            stem=True,
            ent_option=OPTION_GROUP,
            stopwords=OPTION_DELETE,
            **kwargs
    ):
        self.del_diac = del_diac
        self.num_option = num_option
        self.usr_option = usr_option
        self.url_option = url_option
        self.emo_option = emo_option
        self.ent_option = ent_option
        self.hashtag_option = hashtag_option
        self.lc = lc
        self.del_dup = del_dup
        self.del_punc = del_punc
        self.get_conclusion = get_conclusion
        self.token_list = token_list
        self.token_min_filter = token_min_filter
        self.token_max_filter = token_max_filter
        self.tfidf = tfidf
        self.lang = lang
        self.neg = neg
        self.stem = stem
        # self.lang = LangDependency(lang)
        self.stopwords = stopwords

        self.kwargs = {k: v for k, v in kwargs.items() if k[0] != '_'}

        if emo_option == OPTION_NONE:
            self.emo_map = None
        else:
            # self.emo_map = get_compiled_map(os.path.join(os.path.dirname(__file__), 'resources', 'emoticons.json'))
            self.emo_map = EmoticonClassifier()

        docs = [self.tokenize(d) for d in docs]
        self.dictionary = corpora.Dictionary(docs)
        corpus = [self.dictionary.doc2bow(d) for d in docs]
        if self.token_min_filter != 1 or self.token_max_filter != 1.0:
            if self.token_min_filter < 0:
                self.token_min_filter = abs(self.token_min_filter)
            else:
                self.token_min_filter = int(len(corpus) * self.token_min_filter)

            if self.token_max_filter < 0:
                self.token_max_filter = abs(self.token_max_filter)/len(corpus)

            self.dictionary.filter_extremes(no_below=self.token_min_filter, no_above=self.token_max_filter, keep_n=None)

        if self.tfidf:
            self.model = TfidfModel(corpus)
        else:
            self.model = None 

    def __str__(self):
        return "[TextModel {0}]".format(dict(
            num_option=self.num_option,
            usr_option=self.usr_option,
            url_option=self.url_option,
            emo_option=self.emo_option,
            ent_option=self.ent_option,
            hashtag_option=self.hashtag_option,
            lc=self.lc,
            del_dup=self.del_dup,
            del_punc=self.del_punc,
            del_diac=self.del_diac,
            get_conclusion=self.get_conclusion,
            token_list=self.token_list,
            token_min_filter=self.token_min_filter,
            token_max_filter=self.token_max_filter,
            tfidf=self.tfidf,
            lang=self.lang,
            neg=self.neg,
            stem=self.stem,
            stopwords=self.stopwords,
            kwargs=self.kwargs
        ))

    def __getitem__(self, text):
        vec, affinity = self.vectorize(text)
        return vec

    def vectorize(self, text):
        tok = self.tokenize(text)
        bow = self.dictionary.doc2bow(tok)

        if self.tfidf:
            m = self.model[bow]
        else:
            m = bow
        
        try:
            return m, len(bow) / len(tok)
        except ZeroDivisionError:
            return m, 0.0

    def tokenize(self, text):
        if isinstance(text, (list, tuple)):
            tokens = []
            for _text in text:
                tokens.extend(self._tokenize(_text))

            return tokens
        else:
            return self._tokenize(text)

    def _tokenize(self, text):
        if text is None:
            text = ''

        # if self.emo_option == OPTION_DELETE:
        #     text = transform_del(text, self.emo_map)
        # elif self.emo_option == OPTION_GROUP:
        #     text = transform_replace_by_klass(text, self.emo_map)
        if self.emo_map:
            text = self.emo_map.replace(text, option=self.emo_option)

        if self.get_conclusion:
            text = re.sub(r'.+(but|than|then|therefore|however|nonetheless|nevertheless|short of|yet|so)\W', "", text, re.I)

        if self.hashtag_option == OPTION_DELETE:
            text = re.sub(r"#\S+", "", text)
        elif self.hashtag_option == OPTION_GROUP:
            text = re.sub(r"#\S+", "_htag", text)

        # if self.ent_option == OPTION_DELETE:
        #     text = re.sub(r"[A-Z][a-z]+", "", text)
        # elif self.ent_option == OPTION_GROUP:
        #     text = re.sub(r"[A-Z][a-z]+", "_ent", text)

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

        # SMJ 
        #if self.lang:
        #    if self.ent_option:
        #        text = self.lang.process_entities(text, self.ent_option)

        # text = self.lang.transform(text, negation=self.neg, stemming=self.stem, stopwords=self.stopwords)

        # if self.get_conclusion:
        #     a = text.split('.')
        #     while len(a) > 1 and len(a[-1]) == 0:
        #         a.pop()

        #     text = a[-1]

        text = norm_chars(text, del_diac=self.del_diac, del_dup=self.del_dup, del_punc=self.del_punc)

        L = []
        textlist = None

        # _text = memoryview(bytes(text, encoding='utf8'))
        _text = text
        for q in self.token_list:
            if isinstance(q, int):
                if q < 0:
                    if textlist is None:
                        textlist = get_word_list(text)

                    expand_qgrams_word_list(textlist, abs(q), L)
                else:
                    expand_qgrams(_text, q, L)
            else:
                if textlist is None:
                    textlist = get_word_list(text)

                expand_skipgrams_word_list(textlist, q, L)

        # print(len(L), self.token_min_filter)
        return L
