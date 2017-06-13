import json
import re
import os

from collections import defaultdict
from .params import OPTION_DELETE, OPTION_GROUP, OPTION_NONE


# def get_compiled_map(filename):
#     with open(filename) as f:
#         E = json.load(f)

#     X = defaultdict(list)

#     for code, klass in E.items():
#         X[klass].append(re.escape(code))
            
#     Y = {}

#     for klass, codelist in X.items():
#         Y[klass] = re.compile(r"\b{0}\b".format("|".join(codelist)), re.IGNORECASE)
        
#     return Y


# def transform_replace_by_klass(text, map):
#     for klass, regex in map.items():
#         text = regex.sub(" {0} ".format(klass), text)

#     return re.sub(r"\s+", " ", text)


# def transform_del(text, map):
#     for klass, regex in map.items():
#         text = regex.sub(' ', text)

#     return re.sub(r"\s+", " ", text).strip()


class EmoticonClassifier:
    def __init__(self, fname=None):
        if fname is None:
            fname = os.path.join(os.path.dirname(__file__), 'resources', 'emoticons.json')

        self.emolen = defaultdict(dict)
        self.emoreg = []
        self.some = {}

        with open(fname) as f:
            X = json.load(f)

        for c, k in X.items():  # code, klass
            if c.isalpha():
                r = re.compile(r"\b{0}\b".format(c), re.IGNORECASE)
                self.emoreg.append((r, k))
            else:
                self.emolen[len(c)].setdefault(c, k)

            self.some[c[0]] = max(len(c), self.some.get(c[0], 0))

        maxlen = max(self.emolen.keys())
        self.emolen = [self.emolen.get(i, {}) for i in range(maxlen+1)]

    def replace(self, text, option=OPTION_GROUP):
        if option == OPTION_NONE:
            return text

        for pat, klass in self.emoreg:
            if option == OPTION_DELETE:
                klass = ''
 
            text = pat.sub(klass, text)

        T = []
        i = 0
        _text = text.lower()
        while i < len(text):
            replaced = False
            if _text[i] in self.some:
                for lcode in range(1, len(self.emolen)):
                    if i + lcode < len(_text):
                        code = _text[i:i+lcode]
                        klass = self.emolen[lcode].get(code, None)

                        if klass:
                            if option == OPTION_DELETE:
                                klass = ''

                            T.append(klass)
                            replaced = True
                            i += lcode
                            break
            
            if not replaced:
                T.append(text[i])
                i += 1

        return "".join(T)
