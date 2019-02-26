# author: Eric S. Tellez <eric.tellez@infotec.mx>

import os
import sys
import json
import numpy as np
from itertools import combinations


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class Fixed:
    def __init__(self, value):
        self.value = value
        self.valid_values = [value]

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
    def __init__(self, initial_set, max_size=None):
        self.valid_values = []
        if max_size is None:
            max_size = len(initial_set) // 2 + 1

        for i in range(1, len(initial_set)+1):
            for l in combinations(initial_set, i):
                if len(l) <= max_size:
                    self.valid_values.append(l)

    def mismatches(self, value):
        lvalue = len(value)
        for v in self.valid_values:
            # if len(value.intersection(v)) == lvalue - 1 or len(value.union(v)) == lvalue + 1:
            ulen = len(value.union(v))
            ilen = len(value.intersection(v))
            if ulen in (lvalue, lvalue + 1) and ilen in (lvalue, lvalue - 1):
                yield v

    def neighborhood(self, value):
        L = []
        for v in value:
            if isinstance(v, list):
                v = tuple(v)
            L.append(v)

        return list(self.mismatches(set(L)))

    def get_random(self):
        x = len(self.valid_values)
        i = np.random.randint(x)
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


TOKENLIST = [(3, 1), (2, 2), (2, 1), -3, -2, -1, 1, 2, 3, 5, 7, 9]
if "TOKENLIST" in os.environ:
    def _simple_cast(x):
        if isinstance(x, list):
            return tuple(x)
        else:
            return x
    
    TOKENLIST = [_simple_cast(x) for x in json.loads(os.environ["TOKENLIST"])]

MAXTOKENLIST = os.environ.get("MAXTOKENLIST", len(TOKENLIST)//2 + 1)


DefaultParams = dict(
    num_option=Option(),
    usr_option=Option(),
    url_option=Option(),
    emo_option=Option(),

    ent_option=Fixed(OPTION_NONE),
    # hashtag_option=Fixed(OPTION_NONE),
    hashtag_option=Option(),

    select_ent=Fixed(False),
    select_suff=Fixed(False),
    select_conn=Fixed(False),
    
    lc=Boolean(),
    del_dup=Boolean(),
    del_punc=Boolean(),
    del_diac=Boolean(),
    
    token_list=PowersetVariable(TOKENLIST, max_size=MAXTOKENLIST),
    # negative values means for absolute frequencies, positive values between 0 and 1 means for ratio
    token_min_filter=Fixed(-1),
    token_max_filter=Fixed(1.0),
    # token_max_filter=SetVariable([0.5, 0.9, 0.95, 0.99, 1.0]),
    # token_min_filter=SetVariable([-1, -2, -3, -5, -7, -9]),
    weighting=SetVariable(['tfidf', 'tf', 'entropy']),
)

if "PARAMS" in os.environ:
    for k, v in json.loads(os.environ["PARAMS"]).items():
        DefaultParams[k] = Fixed(v)


class ParameterSelection:
    MINIMUM_IMPROVEMENT = 0.001
    IMPROVEMENT_FAILURES = 2

    def __init__(self, params=None):
        if (params is None) or (0 == len(params)):
            params = DefaultParams
        else:
            for k in DefaultParams.keys():
                assert k in params, "{0} is not in given parameters; {1}".format(k, params)

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

    def search(self, fun_score, bsize=32, hill_climbing=True, best_list=None, tabu=None, pool=None):
        if tabu is None:
            tabu = set()
        
        best_list = []
        prev = 0.0
        restarts = 0

        while restarts < ParameterSelection.IMPROVEMENT_FAILURES:
            b = self._search(fun_score, bsize=bsize, hill_climbing=hill_climbing, tabu=tabu, pool=pool)
            best_list.extend(b)
            best_list.sort(key=lambda x: x['_score'], reverse=True)
            curr = best_list[0]['_score']
            if curr - prev < ParameterSelection.MINIMUM_IMPROVEMENT:
                restarts += 1
            
            prev = curr

            print("*** best configuration found (restart failures: {0} of {1})".format(restarts, ParameterSelection.IMPROVEMENT_FAILURES), file=sys.stderr)
            print(json.dumps(best_list[0], sort_keys=True), file=sys.stderr)

        # best_list.sort(key=lambda x: x['_score'], reverse=True)
        return best_list
    
    def _search(self, fun_score, bsize=32, hill_climbing=True, best_list=None, tabu=None, pool=None):
        # initial approximation, montecarlo based procesess

        if tabu is None:
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
                prev = best_list[0]['_score']
                L = []

                for conf in self.expand_neighbors(best_list[0], keywords=keywords):
                    code = get_filename(conf)
                    if code in tabu:
                        continue

                    tabu.add(code)
                    L.append((conf, code))

                if len(L) > bsize:
                    print("Selecting {0} random configuration from a neighborhood of {1}".format(bsize, len(L)), file=sys.stderr)
                    np.random.shuffle(L)
                    L = L[:bsize]
        
                best_list.extend(self.get_best(fun_score, L, desc=desc + " {0}".format(i), pool=pool))
                best_list.sort(key=lambda x: x['_score'], reverse=True)
                curr = best_list[0]['_score']
                if curr - prev < ParameterSelection.MINIMUM_IMPROVEMENT:
                    break

        if hill_climbing:
            print("starting hill climbing with configuration", file=sys.stderr)
            print(json.dumps(best_list[0], sort_keys=True), file=sys.stderr)
            _hill_climbing(['token_list'], "optimizing token_list")
            # _hill_climbing(['token_min_filter', 'token_max_filter'], "optimizing token max and min filters")

            do_vectorizing_opt = len(self.params['token_min_filter'].valid_values) > 1 or len(self.params['token_max_filter'].valid_values) > 1
            if do_vectorizing_opt:
                _hill_climbing(['token_list', 'token_min_filter', 'token_max_filter'], "optimizing all token parameters")

            ks = list(self.params.keys())

            if do_vectorizing_opt:
                ks.remove('token_list')
                ks.remove('token_min_filter')
                ks.remove('token_max_filter')

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
