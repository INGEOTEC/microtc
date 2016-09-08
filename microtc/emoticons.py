import json
import re

from collections import defaultdict


def get_compiled_map(filename):
    with open(filename) as f:
        E = json.load(f)

    X = defaultdict(list)

    for code, klass in E.items():
        X[klass].append(re.escape(code))
            
    Y = {}

    for klass, codelist in X.items():
        Y[klass] = re.compile(r"\b{0}\b".format("|".join(codelist)), re.IGNORECASE)
        
    return Y


def transform_replace_by_klass(text, map):
    for klass, regex in map.items():
        text = regex.sub(" {0} ".format(klass), text)

    return re.sub(r"\s+", " ", text)


def transform_del(text, map):
    for klass, regex in map.items():
        text = regex.sub(' ', text)

    return re.sub(r"\s+", " ", text).strip()
