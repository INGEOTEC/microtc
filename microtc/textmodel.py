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
SKIP_SYMBOLS_AND_SPACES = set(PUNCTUACTION + SYMBOLS + '\t\n\r ')
# SKIP_WORDS = set(["…", "..", "...", "...."])

_SPANISH = 'spanish'
_ENGLISH = 'english'
_ITALIAN = 'italian'
_PORTUGUESE = 'portuguese'
_ARABIC = "arabic"


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
            token_list=[-1],
            token_min_filter=-1,
            token_max_filter=1.0,
            tfidf=True,
            ent_option=OPTION_NONE,
            select_ent=False,
            select_suff=False,
            select_conn=False,
            stem_complement=True,
            lang=None,
            **kwargs
    ):
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
        self.tfidf = tfidf
        self.stem_complement = stem_complement
        self.lang = lang

        self.kwargs = {k: v for k, v in kwargs.items() if k[0] != '_'}

        if emo_option == OPTION_NONE:
            self.emo_map = None
        else:
            # self.emo_map = get_compiled_map(os.path.join(os.path.dirname(__file__), 'resources', 'emoticons.json'))
            self.emo_map = EmoticonClassifier()

        if self.stem_complement:
            if self.lang in [ _SPANISH, _ITALIAN, _PORTUGUESE]:
                from nltk.stem.snowball import SnowballStemmer
                self.stemmer = SnowballStemmer(_SPANISH)
            elif self.lang == _ENGLISH:
                from nltk.stem.porter import PorterStemmer
                self.stemmer =  PorterStemmer()
            elif self.lang == _ARABIC:
                from nltk.stem.isri import ISRIStemmer
                self.stemmer = ISRIStemmer()
        else:
            self.stemmer = None

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

    def stemming_complement(self, text):
        if self.stemmer is None:
            print("Stemmer is not initialized") 
            import sys
            sys.exit(-1)

        tokens = text.split()
        stem_c = []
        for t in tokens:
            if not (t.startswith("http://") or  
                t.startswith("ftp://") or
                t.startswith("@") or
                t.startswith("#")):
                s = self.stemmer.stem(t)
                c = t.replace(s,"")
                stem_c.append(c)            
            else:
                stem_c.append(t)
        return " ".join(stem_c)

    def _tokenize(self, text):
        if text is None:
            text = ''

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

        text = norm_chars(text, del_diac=self.del_diac, del_dup=self.del_dup, del_punc=self.del_punc)

        L = []
        textlist = None

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

        if self.select_suff:
            L = [tok for tok in L if tok[-1] in SKIP_SYMBOLS_AND_SPACES]
            if len(L) == 0:
                L = ['~']
            
        if self.select_conn:
            L = [tok for tok in L if '~' in tok and tok[0] != '~' and tok[-1] != '~']
            if len(L) == 0:
                L = ['~']

        return L
