# author: Eric S. Tellez <eric.tellez@infotec.mx>


import numpy as np
import logging
from itertools import combinations


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')

class Fixed:
    def __init__(self, value):
        self.value = value

    def neighborhood(self, v):
        return []

    def get_random(self):
        return self.value


class SetVariable:
    def __init__(self, values):
        self.valid_values = list(values)

    def neighborhood(self, value):
        return [u for u in self.valid_values if u != value]

    def get_random(self):
        i = np.random.randint(len(self.valid_values))
        return self.valid_values[i]

class PowersetVariable:
    def __init__(self, initial_set):
        self.valid_values = []

        for i in range(1, len(initial_set)+1):
            self.valid_values.extend(combinations(initial_set, i))

    def mismatches(self, value):
        lvalue = len(value)
        for v in self.valid_values:
            # if len(value.intersection(v)) == lvalue - 1 or len(value.union(v)) == lvalue + 1:
            ulen = len(value.union(v))
            ilen = len(value.intersection(v))
            if ulen in (lvalue, lvalue + 1) and ilen in (lvalue, lvalue - 1):
                yield v

    def neighborhood(self, value):
        return list(self.mismatches(set(value)))

    def get_random(self):
        i = np.random.randint(len(self.valid_values))
        return self.valid_values[i]


OPTION_NONE = 'none'
OPTION_GROUP = 'group'
OPTION_DELETE = 'delete'
BASIC_OPTIONS = [OPTION_DELETE, OPTION_GROUP, OPTION_NONE]


def Option():
    return SetVariable(BASIC_OPTIONS)


def Uniform(left, right, k=10):
    d = (right - left) * np.random.random_sample(k) + left
    return SetVariable(d)


def Normal(mean, sigma, k=10):
    d = mean * np.random.randn(k) + sigma
    return SetVariable(d)


def Boolean():
    return SetVariable([False, True])


DefaultParams = dict(
    strip_diac=Boolean(),
    num_option=Option(),
    usr_option=Fixed(OPTION_NONE),
    url_option=Fixed(OPTION_NONE),
    emo_option=Option(),
    lc=Boolean(),
    del_dup1=Boolean(),
    del_punc=Boolean(),
    token_list=PowersetVariable([(2, 2), (2, 1), -3, -2, -1, 1, 2, 3, 5, 7]),
    # token_list=PowersetVariable([-3, -2, -1, 1, 2, 3, 5, 7]),
    negation=Fixed(False),
    stemming=Fixed(False),
    stopwords=Fixed(OPTION_NONE),
    lang=Fixed(None),
)


class ParameterSelection:
    def __init__(self, params=None):
        if params is None:
            params = DefaultParams

        self.params = params

    def sample_param_space(self, n):
        for i in range(n):
            kwargs = {}
            for k, v in self.params.items():
                kwargs[k] = v.get_random()

            yield kwargs

    def expand_neighbors(self, s, keywords=None):
        if keywords is None:
            keywords = set(s.keys())

        for k, v in sorted(s.items()):
            if k[0] == '_' or k not in keywords:
                # by convention, metadata starts with underscore
                continue

            vtype = self.params[k]
            if isinstance(vtype, Fixed):
                continue

            for neighbor in vtype.neighborhood(v):
                x = s.copy()
                x[k] = neighbor
                yield(x)

    def get_best(self, fun_score, cand, desc="searching for params", pool=None):
        if pool is None:
            # X = list(map(fun_score, cand))
            X = [fun_score(x) for x in tqdm(cand, desc=desc, total=len(cand))]
        else:
            # X = list(pool.map(fun_score, cand))
            X = [x for x in tqdm(pool.imap_unordered(fun_score, cand), desc=desc, total=len(cand))]

        # a list of tuples (score, conf)
        X.sort(key=lambda x: x['_score'], reverse=True)
        return X

    def search(self, fun_score, bsize=32, hill_climbing=True, pool=None, best_list=None):
        # initial approximation, montecarlo based procesess

        tabu = set()  # memory for tabu search

        if best_list is None:
            L = []
            for conf in self.sample_param_space(bsize):
                code = get_filename(conf)
                if code in tabu:
                    continue

                tabu.add(code)
                L.append((conf, code))
            
            best_list = self.get_best(fun_score, L, pool=pool)
        else:
            for conf in best_list:
                tabu.add(get_filename(conf))

        def _hill_climbing(keywords, desc):
            # second approximation, a hill climbing process
            i = 0
            while True:
                i += 1
                bscore = best_list[0]['_score']
                L = []

                for conf in self.expand_neighbors(best_list[0], keywords=keywords):
                    code = get_filename(conf)
                    if code in tabu:
                        continue

                    tabu.add(code)
                    L.append((conf, code))

                best_list.extend(self.get_best(fun_score, L, desc=desc + " {0}".format(i), pool=pool))
                best_list.sort(key=lambda x: x['_score'], reverse=True)
                if bscore == best_list[0]['_score']:
                    break

        if hill_climbing:
            _hill_climbing(['token_list'], "optimizing token_list")
            ks = list(self.params.keys())
            ks.remove('token_list')
            _hill_climbing(ks, "optimizing the rest of params")

        return best_list


def get_filename(kwargs, basename=None):
    L = []
    if basename:
        L.append(basename)
        
    for k, v in sorted(kwargs.items()):
        if k[0] == '_':
            continue

        L.append("{0}={1}".format(k, v).replace(" ", ""))

    return "-".join(L)
