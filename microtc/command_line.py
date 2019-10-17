# Copyright 2016 Mario Graff (https://github.com/mgraffg)
# with contributions of Eric S. Tellez <eric.tellez@infotec.mx>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import microtc
import gzip
from microtc.wrappers import ClassifierWrapper, RegressorWrapper
from microtc.utils import read_data, read_data_labels, read_data_values, tweet_iterator
from multiprocessing import cpu_count, Pool
from collections import defaultdict
from .params import ParameterSelection
from .scorewrapper import ScoreKFoldWrapper, ScoreSampleWrapper
from .regscorewrapper import RegressionScoreKFoldWrapper, RegressionScoreSampleWrapper
from .textmodel import TextModel
from .utils import KLASS, VALUE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np
import os
import json
from .utils import save_model, load_model


def load_json(filename):
    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename)
        X = json.load(f)
        f.close()
        return X
    else:
        with open(filename) as f:
            return json.load(f)


def balance(X, y):
    C = defaultdict(list)
    for x, label in zip(X, y):
        C[label].append(x)

    size, label = min(map(lambda label: (len(C[label]), label), C.keys()), key=lambda x: x[0])
    for label, xlist in C.items():
        while len(xlist) > size:
            last = xlist.pop()
            i = np.random.randint(0, size)
            if isinstance(xlist[i], list):
                xlist[i] = last
            else:
                xlist[i] = [xlist[i], last]

    X = []
    y = []
    for label, xlist in C.items():
        X.extend(xlist)
        for i in range(len(xlist)):
            y.append(label)

    return X, y


def clean_params(kw):
    params = TextModel.params()
    return {k: v for k, v in kw.items() if k in params}


class CommandLine(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='microtc')
        self.training_set()
        self.predict_kfold()
        self.param_set()
        self.param_search()
        self.version()

    def version(self):
        pa = self.parser.add_argument
        pa('--version',
           action='version', version='microtc %s' % microtc.__version__)

    def predict_kfold(self):
        pa = self.parser.add_argument
        pa('-k', '--kratio', dest='kratio',
           help='Predict the training set using stratified k-fold (k > 1) or a sampling ratio (when 0 < k < 1)',
           default="0.8",
           type=str)

    def training_set(self):
        cdn = 'File containing the training set'
        pa = self.parser.add_argument
        pa('training_set',
           nargs='+',
           default=None,
           help=cdn)

    def param_search(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int, default=32, help="The sample size of the parameter")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False, action='store_true',
            help="Determines if hillclimbing search is also perfomed to improve the selection of parameters")
        pa('-r', '--resume', dest='best_list', default=None, help="Loads the given file and resumes the search process")
        pa('-b', '--balanced', dest='balanced', default=False,
           action='store_true',
           help="Artificially balances the dataset to reduce bias in unbalanced number of examples per class (only works for classification)")
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1, help="Number of processes to compute the best setup")
        pa('-S', '--score', dest='score', type=str, default='macrof1',
           help="The name of the score to be optimized (classification scores: {0}); (regression scores: {1}) it defaults to macrof1".format(
               ScoreSampleWrapper.valid_scores,
               RegressionScoreSampleWrapper.valid_scores
           ))
        pa('--conf', dest='conf', type=str, default=None, help="Do not perform search, just evaluate the given configuration (in json-format)")

    def param_set(self):
        pa = self.parser.add_argument
        pa('-o', '--output-file', dest='output',
           help='File name to store the output')
        pa('--seed', default=0, type=int)

    def get_output(self):
        if self.data.output is None:
            return self.data.training_set[0] + ".output"
        return self.data.output

    def main(self, args=None, params=None):
        self.data = self.parser.parse_args(args=args)
        np.random.seed(self.data.seed)
        if self.data.numprocs == 1:
            pool = None
        elif self.data.numprocs == 0:
            pool = Pool(cpu_count())
        else:
            pool = Pool(self.data.numprocs)

        assert self.data.score.split(":")[0] in ScoreSampleWrapper.valid_scores + RegressionScoreSampleWrapper.valid_scores, "Unknown score {0}".format(self.data.score)
        if self.data.score in RegressionScoreSampleWrapper.valid_scores:
            ScoreSample = RegressionScoreSampleWrapper
            ScoreKFold = RegressionScoreKFoldWrapper
            _read_data = read_data_values
        else:
            ScoreSample = ScoreSampleWrapper
            ScoreKFold = ScoreKFoldWrapper
            _read_data = read_data_labels

        sel = ParameterSelection(params=params)
        X, y = [], []
        Xstatic, ystatic = [], []
        for train in self.data.training_set:
            if train.startswith("static:"):
                X_, y_ = _read_data(train[7:])
                Xstatic.extend([x for x in tweet_iterator(train[7:])])
                ystatic.extend(y_)
            else:
                X_, y_ = _read_data(train)
                X.extend([x for x in tweet_iterator(train)])
                y.extend(y_)

        if self.data.balanced:
            X, y = balance(X, y)

        if ":" in self.data.kratio:
            ratio, test_ratio = self.data.kratio.split(":")
            fun_score = ScoreSample(X, y, Xstatic=Xstatic, ystatic=ystatic, ratio=float(ratio), test_ratio=float(test_ratio), score=self.data.score, random_state=self.data.seed)
        else:
            ratio = float(self.data.kratio)
            if ratio == 1.0:
                raise ValueError('k=1 is undefined')
            if ratio > 1:
                fun_score = ScoreKFold(X, y, Xstatic=Xstatic, ystatic=ystatic, nfolds=int(ratio), score=self.data.score, random_state=self.data.seed)
            else:
                fun_score = ScoreSample(X, y, Xstatic=Xstatic, ystatic=ystatic, ratio=ratio, score=self.data.score, random_state=self.data.seed)

        if self.data.best_list:
            best_list = load_json(self.data.best_list)
        else:
            best_list = None

        if self.data.conf:
            conf = json.loads(self.data.conf)
            best_list = sel.get_best(fun_score, (conf, 'direct-input'))
        else:
            best_list = sel.search(
                fun_score,
                bsize=self.data.samplesize,
                hill_climbing=self.data.hill_climbing,
                pool=pool,
                best_list=best_list
            )

        best_list0 = list(filter(lambda x: '_error' not in x, best_list))
        if len(best_list0) == 0:
            raise Exception("ERROR best_list is empty" + repr(best_list))

        best_list = best_list0
        with open(self.get_output(), 'w') as fpt:
            fpt.write(json.dumps(best_list0, indent=2, sort_keys=True))

        return best_list0


class CommandLineTrain(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='microtc')
        self.param_set()
        self.training_set()
        self.param_train()
        self.version()

    def param_train(self):
        pa = self.parser.add_argument
        pa('-m', '--model-params', dest='params_fname', type=str, required=True,
           help="TextModel params")
        pa('-i', '--i-th', dest='position', type=int, default=0,  # best by default
           help="i-th model in the set of configurations (defaults to the best, i.e., 0)")
        pa('-l', '--labels', dest='labels', type=str,
           help="a comma separated list of valid labels")
        pa('-b', '--balanced', dest='balanced', default=False, action='store_true',
           help="Artificially balances the dataset to reduce bias in unbalanced number of examples per class")
        pa('--conf', dest='conf', type=str,
           help="Specifies the configuration in JSON-format")
        pa('-R', '--regression', dest='regression', action='store_true',
           help="The model will be a regressor")

    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        if self.data.conf:
            best = json.loads(self.data.conf)
        else:
            best = load_json(self.data.params_fname)[self.data.position]
        best = clean_params(best)
        if self.data.regression:
            _read_data = read_data_values
            wrapper = RegressorWrapper
        else:
            _read_data = read_data_labels
            wrapper = ClassifierWrapper

        corpus, values = [], []
        for train in self.data.training_set:
            X_, y_ = _read_data(train)
            corpus.extend([x for x in tweet_iterator(train)])
            values.extend(y_)

        if self.data.balanced:
            corpus, values = balance(corpus, values)

        t = TextModel(corpus, **best)
        if self.data.regression:
            le = None
            y = values
        else:
            le = LabelEncoder()
            if self.data.labels:
                le.fit(self.data.labels.split(','))
            else:
                le.fit(values)

            y = le.transform(values)

        c = wrapper()
        X = [t[x] for x in corpus]
        c.fit(X, y)
        save_model([t, c, le], self.get_output())
        return [t, c, le]


class CommandLinePredict(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='microtc')
        self.param_set()
        self.training_set()
        self.param_test()
        self.version()

    def param_test(self):
        pa = self.parser.add_argument
        pa('-m', '--model', dest='model', type=str,
           required=True,
           help="SVM Model file name")

    def get_output(self):
        if self.data.output is None:
            return self.data.test_set[0] + ".predicted"

        return self.data.output

    def training_set(self):
        cdn = 'File containing the test set'
        pa = self.parser.add_argument
        pa('test_set',
           default=None,
           help=cdn)
        pa('--ordinal', dest='ordinal', default=None,
           help="rounds a regression prediction to the nearest integer among the given start:end range")

    def main(self, args=None, model_svc_le=None):
        self.data = self.parser.parse_args(args=args)
        if model_svc_le is None:
            model, svc, le = load_model(self.data.model)
        else:
            model, svc, le = model_svc_le

        veclist = []
        for x in read_data(self.data.test_set):
            v = model[x]
            veclist.append(v)

        L = []
        if le is None:
            hy = svc.predict(veclist)

            if self.data.ordinal:
                start, end = self.data.ordinal.split(':')
                start = int(start)
                end = int(end)

                for tweet, pred in zip(tweet_iterator(self.data.test_set), hy):
                    c = round(pred)
                    if c < start:
                        c = start
                    elif c > end:
                        c = end

                    if c == 0:  # handles IEEE's negative cero -0.0
                        c = 0

                    tweet[VALUE] = int(c)
                    L.append(tweet)
            else:
                for tweet, pred in zip(tweet_iterator(self.data.test_set), hy):
                    tweet[VALUE] = pred
                    L.append(tweet)
        else:
            decision_function = None
            predict_proba = None
            try:
                decision_function = svc.decision_function(veclist).tolist()
            except AttributeError:
                try:
                    predict_proba = svc.predict_proba(veclist).tolist()
                except AttributeError:
                    pass

            hyy = le.inverse_transform(svc.predict(veclist))

            for i, tweet in enumerate(tweet_iterator(self.data.test_set)):
                if decision_function is not None:
                    tweet['decision_function'] = decision_function[i]
                if predict_proba is not None:
                    tweet['predict_proba'] = predict_proba[i]

                klass = hyy[i]
                tweet[KLASS] = str(klass)
                tweet['predicted'] = tweet[KLASS]
                L.append(tweet)

        with open(self.get_output(), 'w') as fpt:
            for tweet in L:
                fpt.write(json.dumps(tweet)+"\n")

        return L


class CommandLineTextModel(CommandLinePredict):
    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        textmodel, svc, le = load_model(self.data.model)
        L = []
        with open(self.get_output(), 'w') as fpt:
            for tw in tweet_iterator(self.data.test_set):
                tw["vec"] = [[int(a), float(b)] for a, b in textmodel[tw['text']]]
                tw["vecsize"] = svc.num_terms
                L.append(tw)
                fpt.write(json.dumps(tw) + "\n")
        return L


class CommandLineRetrain(CommandLinePredict):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='microtc')
        pa = self.parser.add_argument
        pa('-m', '--model', default=None, type=str,
           dest='model', help='specifies the input model')
        pa('-o', '--output', default='', type=str, dest='output',
           help='save the modified model to the given output file')
        pa('training_set', default=None, nargs='+', help="The trainset, it can be used to retrain a model or readjust the model")
        pa('-R', '--regression', dest='regression', action='store_true', help="The model will be a regressor")

    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        # logging.basicConfig(level=self.data.verbose)
        textmodel, svc, le = load_model(self.data.model)

        if self.data.regression:
            _read_data = read_data_values
            wrapper = RegressorWrapper
        else:
            _read_data = read_data_labels
            wrapper = ClassifierWrapper

        corpus, values = [], []
        for train in self.data.training_set:
            X_, y_ = _read_data(train)
            corpus.extend(X_)
            values.extend(y_)

        if self.data.regression:
            le = None
            y = values
        else:
            y = le.transform(values)

        c = wrapper()
        X = [textmodel[x] for x in corpus]
        c.fit(X, y)
        save_model([textmodel, c, le], self.get_output())
        return [textmodel, c, le]


class CommandLineKfolds(CommandLineTrain):
    def __init__(self):
        super(CommandLineKfolds, self).__init__()
        self.param_kfold()

    def param_kfold(self):
        pa = self.parser.add_argument
        pa('--update-klass', default=False, dest='update_klass',
           action="store_true", help='Indicates whether the klass should be updated (default False)')
        pa('-k', '--kratio', dest='kratio',
           help='Predict the training set using k-fold (k > 1)',
           default="5",
           type=int)

    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        assert not self.data.update_klass
        if self.data.conf:
            best = json.loads(self.data.conf)
        else:
            best = load_json(self.data.params_fname)[0]
        best = clean_params(best)
        corpus, labels = [], []
        for train in self.data.training_set:
            X_, y_ = read_data_labels(train)
            corpus.extend([x for x in tweet_iterator(train)])
            labels.extend(y_)
        le = LabelEncoder()
        if self.data.labels:
            le.fit(self.data.labels.split(','))
        else:
            le.fit(labels)
        y = le.transform(labels)
        model_klasses = os.environ.get('TEXTMODEL_KLASSES')

        if model_klasses:
            model_klasses = le.transform(model_klasses.split(','))
            docs_ = []
            labels_ = []
            for i in range(len(corpus)):
                if y[i] in model_klasses:
                    docs_.append(corpus[i])
                    labels_.append(y[i])

            t = TextModel(docs_, **best)
        else:
            t = TextModel(corpus, **best)

        X = [t[x] for x in corpus]
        hy = [None for x in y]
        for tr, ts in KFold(n_splits=self.data.kratio,
                            shuffle=True, random_state=self.data.seed).split(X):
            c = ClassifierWrapper()
            c.fit([X[x] for x in tr], [y[x] for x in tr])
            _ = c.decision_function([X[x] for x in ts])
            [hy.__setitem__(k, v) for k, v in zip(ts, _)]

        i = 0
        with open(self.get_output(), 'w') as fpt:
            for train in self.data.training_set:
                for tweet in tweet_iterator(train):
                    tweet['decision_function'] = hy[i].tolist()
                    i += 1
                    fpt.write(json.dumps(tweet)+"\n")
        return hy


def params(*args, **kwargs):
    c = CommandLine()
    if len(args) == 0:
        args = None
    return c.main(args, params=kwargs)


def train(*args, **kwargs):
    c = CommandLineTrain()
    if len(args) == 0:
        args = None
    return c.main(args, **kwargs)


def retrain(*args, **kwargs):
    c = CommandLineRetrain()
    if len(args) == 0:
        args = None
    return c.main(args, **kwargs)


def predict(*args, **kwargs):
    c = CommandLinePredict()
    if len(args) == 0:
        args = None
    return c.main(args, **kwargs)


def textmodel(*args, **kwargs):
    c = CommandLineTextModel()
    if len(args) == 0:
        args = None
    return c.main(args, **kwargs)


def kfolds(*args, **kwargs):
    c = CommandLineKfolds()
    if len(args) == 0:
        args = None
    return c.main(args, **kwargs)
