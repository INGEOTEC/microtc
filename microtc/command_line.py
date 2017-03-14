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
import logging
import microtc
from microtc.classifier import ClassifierWrapper
from microtc.utils import read_data, tweet_iterator
# from microtc.params import OPTION_DELETE
from multiprocessing import cpu_count, Pool
from .params import ParameterSelection
from .scorewrapper import ScoreKFoldWrapper, ScoreSampleWrapper
from .utils import read_data_labels
from .textmodel import TextModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import json
import pickle

# from microtc.params import ParameterSelection


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
        pa('--verbose', dest='verbose', type=int,
           help='Logging level default: INFO + 1',
           default=logging.INFO+1)

    def param_search(self):
        pa = self.parser.add_argument
        pa('-s', '--sample', dest='samplesize', type=int,
           default=32,
           help="The sample size of the parameter")
        pa('-H', '--hillclimbing', dest='hill_climbing', default=False,
           action='store_true',
           help="Determines if hillclimbing search is also perfomed to improve the selection of parameters")
        pa('-r', '--resume', dest='best_list', default=None,
           help="Loads the given file and resumes the search process")
        pa('-n', '--numprocs', dest='numprocs', type=int, default=1,
           help="Number of processes to compute the best setup")
        pa('-S', '--score', dest='score', type=str, default='macrof1',
           help="The name of the score to be optimized (macrof1|weightedf1|accuracy|avgf1:klass1:klass2); it defaults to macrof1")

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
        logging.basicConfig(level=self.data.verbose)
        if self.data.numprocs == 1:
            pool = None
        elif self.data.numprocs == 0:
            pool = Pool(cpu_count())
        else:
            pool = Pool(self.data.numprocs)

        assert self.data.score.split(":")[0] in ('macrorecall', 'macrof1', 'microf1', 'weightedf1', 'accuracy', 'avgf1'), "Unknown score {0}".format(self.data.score)

        sel = ParameterSelection(params=params)

        X, y = [], []
        Xstatic, ystatic = [], []
        for train in self.data.training_set:
            if train.startswith("static:"):
                X_, y_ = read_data_labels(train[7:])
                Xstatic.extend(X_)
                ystatic.extend(y_)
            else:
                X_, y_ = read_data_labels(train)
                X.extend(X_)
                y.extend(y_)

        if ":" in self.data.kratio:
            ratio, test_ratio = self.data.kratio.split(":")
            fun_score = ScoreSampleWrapper(X, y,
                                           Xstatic=Xstatic, ystatic=ystatic,
                                           ratio=float(ratio), test_ratio=float(test_ratio), score=self.data.score, random_state=self.data.seed)
        else:
            ratio = float(self.data.kratio)
            if ratio == 1.0:
                raise ValueError('k=1 is undefined')

            if ratio > 1:
                fun_score = ScoreKFoldWrapper(X, y, Xstatic=Xstatic, ystatic=ystatic, nfolds=int(ratio), score=self.data.score, random_state=self.data.seed)
            else:
                fun_score = ScoreSampleWrapper(X, y, Xstatic=Xstatic, ystatic=ystatic, ratio=ratio, score=self.data.score, random_state=self.data.seed)

        if self.data.best_list:
            with open(self.data.best_list) as f:
                best_list = json.load(f)
        else:
            best_list = None

        best_list = sel.search(
            fun_score,
            bsize=self.data.samplesize,
            hill_climbing=self.data.hill_climbing,
            pool=pool,
            best_list=best_list
        )

        with open(self.get_output(), 'w') as fpt:
            fpt.write(json.dumps(best_list, indent=2, sort_keys=True))

        return best_list


class CommandLineTrain(CommandLine):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='microtc')
        self.param_set()
        self.training_set()
        self.param_train()
        self.version()

    def param_train(self):
        pa = self.parser.add_argument
        pa('-m', '--model-params', dest='params_fname', type=str,
           required=True,
           help="TextModel params")
        pa('-l', '--labels', dest='labels', type=str,
           help="a comma separated list of valid labels")

    def main(self, args=None):
        self.data = self.parser.parse_args(args=args)
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.params_fname) as fpt:
            param_list = json.loads(fpt.read())

        corpus, labels = [], []
        for train in self.data.training_set:
            X_, y_ = read_data_labels(train)
            corpus.extend(X_)
            labels.extend(y_)

        best = param_list[0]
        t = TextModel(corpus, **best)
        le = LabelEncoder()
        if self.data.labels:
            le.fit(self.data.labels.split(','))
        else:
            le.fit(labels)
        y = le.transform(labels)
        c = ClassifierWrapper()
        X = [t[x] for x in corpus]
        c.fit(X, y)
        
        with open(self.get_output(), 'wb') as fpt:
            pickle.dump([t, c, le], fpt)

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

    def training_set(self):
        cdn = 'File containing the test set'
        pa = self.parser.add_argument
        pa('test_set',
           default=None,
           help=cdn)
        pa('--verbose', dest='verbose', type=int,
           help='Logging level default: INFO + 1',
           default=logging.INFO+1)

    def main(self, args=None, model_svc_le=None):
        self.data = self.parser.parse_args(args=args)
        logging.basicConfig(level=self.data.verbose)
        if model_svc_le is None:
            with open(self.data.model, 'rb') as fpt:
                model, svc, le = pickle.load(fpt)
        else:
            model, svc, le = model_svc_le

        veclist, afflist = [], []
        for x in read_data(self.data.test_set):
            v, a = model.vectorize(x)
            veclist.append(v)
            afflist.append(a)

        L = []
        hy = svc.decision_function(veclist)
        hyy = le.inverse_transform(svc.predict(veclist))
        KLASS = os.environ.get('KLASS', 'klass')
        for tweet, scores, klass, aff in zip(tweet_iterator(self.data.test_set), hy, hyy, afflist):
            # if True:
            #     print("-YY>", scores)
            #     print("-XX>", scores.shape, len(scores.shape))
            #     print(svc.svc.classes_)
            #     print(le)

            # if len(scores.shape) == 0:
            #     index = 0 if scores < 0.0 else 1
            # elif len(scores.shape) == 1:
            #     index = np.argmax(scores)
            # else:
            #     index = scores.argmax(axis=1)

            # klass = le.inverse_transform(svc.svc.classes_[index])
            
            tweet['decision_function'] = scores.tolist()
            tweet['voc_affinity'] = aff
            tweet[KLASS] = str(klass)
            L.append(tweet)

        with open(self.get_output(), 'w') as fpt:
            for tweet in L:
                fpt.write(json.dumps(tweet)+"\n")

        return L


class CommandLineTextModel(CommandLinePredict):
    def main(self):
        self.data = self.parser.parse_args()
        logging.basicConfig(level=self.data.verbose)
        with open(self.data.model, 'rb') as fpt:
            textmodel, svc, le = pickle.load(fpt)

        L = []
        with open(self.get_output(), 'w') as fpt:
            for tw in tweet_iterator(self.data.test_set):
                extra = dict(textmodel[tw['text']] + [('num_terms', svc.num_terms)])
                tw.update(extra)
                L.append(tw)
                fpt.write(json.dumps(tw) + "\n")

        return L


def params(*args, **kwargs):
    c = CommandLine()
    return c.main(*args, **kwargs)


def train(*args, **kwargs):
    c = CommandLineTrain()
    return c.main(*args, **kwargs)


def predict(*args, **kwargs):
    c = CommandLinePredict()
    return c.main(*args, **kwargs)


def textmodel(*args, **kwargs):
    c = CommandLineTextModel()
    return c.main(*args, **kwargs)

