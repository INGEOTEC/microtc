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


def convert_emoji(emoji):
    """Convert the code points into characters

    :param emoji: code point
    :return: emoji
    :rtype: str
    """
    if emoji.count(".."):
        init, end = [int(x, base=16) for x in emoji.split("..")]
        return [chr(x) for x in range(init, end + 1)]
    emojis = emoji.split()
    if len(emojis) > 1:
        emojis = filter(len, [x.strip() for x in emojis])
        return "".join([convert_emoji(x) for x in emojis])
    return chr(int(emoji, base=16))

    
def read_emoji_standard(fname, emos=None):
    """Read the emoji standard files

    :param fname: Path to the file
    :type fname: str
    :param emos: Dictionary computed from a previous called
    :type emos: dict
    :return: Emoji with types as dictionary 
    :rtype: dict
    """

    emos = dict() if emos is None else emos
    with open(fname, encoding="utf-8") as fpt:
        for line in fpt.readlines():
            try:
                line = line[:line.index("#")].strip()            
                if len(line) == 0:
                    continue                
                value, tipo = [x.strip() for x in line.split(";")][:2]
            except ValueError:
                continue
            value = convert_emoji(value)
            values = value if isinstance(value, list) else [value]
            for value in values:
                lst = emos.get(value, list())
                lst.append(tipo)
                emos[value] = lst
    return emos


def create_data_structure(emojis):
    """Create data structure to store the emojis
    :param emojis: Dictionary of emoji
    :type emojis: dict
    :rtype: dict
    """

    head = dict()
    for word in emojis.keys():
        current = head
        for char in word:
            try:
                current = current[char]
            except KeyError:
                _ = dict()
                current[char] = _
                current = _
        current["__end__"] = True
    return head


def find_emoji(data, text):
    """Test whether text has an emoji

    :param data: Output of :py:func:`create_data_structure`
    :type data: dict
    :return: list of pairs, init and end of the word
    :rtype: list
    """

    blocks = list()
    init = i = end = 0
    head = data
    current = head
    while i < len(text):
        char = text[i]
        try:
            current = current[char]
            i += 1
            if "__end__" in current:
                end = i
        except KeyError:
            current = head
            if end > init:
                blocks.append([init, end])
                init = i = end
            elif i > init:
                init = end = i
            else:
                init += 1
                i = end = init
    if end > init:
        blocks.append([init, end])                
    return blocks
