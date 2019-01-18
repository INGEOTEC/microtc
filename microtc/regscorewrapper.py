# Copyright 2016 Mario Graff (https://github.com/mgraffg)
# Copyright 2016 Eric S. Tellez <eric.tellez@infotec.mx>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from time import time
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn import preprocessing
from sklearn import model_selection
from microtc.textmodel import TextModel
from microtc.wrappers import RegressorWrapper


class RegressionScoreSampleWrapper(object):
    valid_scores = ['r2', 'pearsonr', 'spearmanr']

    def __init__(self, X, y, Xstatic=[], ystatic=[], ratio=0.8, test_ratio=None, score='r2', classifier=RegressorWrapper, random_state=None):
        assert ratio < 1, "ratio {0} is invalid, valid values are 0 < ratio < 1".format(ratio)
        self.score = score
        self.le = preprocessing.LabelEncoder().fit(y)
        self.create_classifier = classifier
        if test_ratio is None:
            test_ratio = 1.0 - ratio

        I = list(range(len(y)))
        np.random.shuffle(I)
        s = int(np.ceil(len(y) * ratio))
        s_end = int(np.ceil(len(y) * test_ratio))
        y = self.le.transform(y)
        train, test = I[:s], I[s:s+s_end]
        self.train_corpus = [X[i] for i in train]
        self.train_corpus.extend(Xstatic)

        if len(ystatic) > 0:
            ystatic = self.le.transform(ystatic)
            self.train_y = np.hstack((y[train], ystatic))
        else:
            self.train_y = y[train]

        self.test_corpus = [X[i] for i in test]
        self.test_y = y[test]

    def __call__(self, conf_code):
        conf, code = conf_code
        st = time()
        textmodel = TextModel(self.train_corpus, **conf)
        train_X = [textmodel[doc] for doc in self.train_corpus]
        c = self.create_classifier()
        # c.fit(train_X, self.train_y)
        try:
            c.fit(train_X, self.train_y)
        except ValueError:
            conf["_error"] = "this configuration produces an empty matrix"
            conf["_score"] = 0.0
            return conf
    
        test_X = [textmodel[doc] for doc in self.test_corpus]
        pred_y = c.predict(test_X)
        self.compute_score(conf, pred_y)
        conf['_time'] = (time() - st)
        return conf

    def compute_score(self, conf, hy):
        conf['_r2'] = r2_score(self.test_y, hy)
        conf['_spearmanr'] = spearmanr(self.test_y, hy)[0]
        conf['_pearsonr'] = pearsonr(self.test_y, hy)[0]
        conf['_score'] = conf['_' + self.score]
        # print(conf)


class RegressionScoreKFoldWrapper(RegressionScoreSampleWrapper):
    def __init__(self, X, y, Xstatic=[], ystatic=[], nfolds=5, score='r2', classifier=RegressorWrapper, random_state=None):
        self.nfolds = nfolds
        self.score = score
        # self.X = np.array(X)
        self.X = X
        self.Xstatic = Xstatic
        self.le = preprocessing.LabelEncoder().fit(y)
        self.y = self.le.transform(y)
        if len(ystatic) > 0:
            self.ystatic = self.le.transform(ystatic)
        else:
            self.ystatic = []
        self.test_y = self.y
        self.create_classifier = classifier
        self.kfolds = model_selection.KFold(n_splits=nfolds, shuffle=True, random_state=random_state)

    def __call__(self, conf_code):
        conf, code = conf_code
        st = time()
        predY = np.zeros(len(self.y))
        # X = np.array(self.X)
        for train, test in self.kfolds.split(self.X):
            # A = X[train]
            A = [self.X[i] for i in train]
            if len(self.Xstatic) > 0:
                A.extend(self.Xstatic)

            trainY = self.y[train]
            if len(self.ystatic) > 0:
                trainY = np.hstack((trainY, self.ystatic))

            textmodel = TextModel(A, **conf)
            trainX = [textmodel[x] for x in A]

            c = self.create_classifier()
            try:
                c.fit(trainX, trainY)
            except ValueError:
                conf["_error"] = "this configuration produces an empty matrix"
                conf["_score"] = 0.0
                return conf

            testX = [textmodel[self.X[i]] for i in test]
            predY[test] = c.predict(testX)

        self.compute_score(conf, predY)
        conf['_time'] = (time() - st) / self.nfolds
        return conf
