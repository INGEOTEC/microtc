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
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn import preprocessing
from sklearn import cross_validation
from microtc.textmodel import TextModel
from microtc.classifier import ClassifierWrapper


class ScoreSampleWrapper(object):
    def __init__(self, X, y, ratio=0.8, score='macrof1', classifier=ClassifierWrapper, random_state=None):
        self.score = score
        self.le = preprocessing.LabelEncoder().fit(y)
        self.create_classifier = classifier
        I = list(range(len(y)))
        np.random.shuffle(I)
        s = int(np.ceil(len(y) * ratio))
        y = np.array(self.le.transform(y))
        train, test = I[:s], I[s:]
        self.train_corpus = [X[i] for i in train]
        self.train_y = y[train]
        self.test_corpus = [X[i] for i in test]
        self.test_y = y[test]

    def __call__(self, conf_code):
        conf, code = conf_code
        st = time()
        textmodel = TextModel(self.train_corpus, **conf)
        train_X = [textmodel[doc] for doc in self.train_corpus]
        c = self.create_classifier()
        c.fit(train_X, self.train_y)
        test_X = [textmodel[doc] for doc in self.test_corpus]
        pred_y = c.predict(test_X)
        self.compute_score(conf, pred_y)
        conf['_time'] = (time() - st)
        return conf

    def compute_score(self, conf, hy):
        conf['_all_f1'] = M = {self.le.inverse_transform([klass])[0]: f1 for klass, f1 in enumerate(f1_score(self.test_y, hy, average=None))}
        conf['_all_recall'] = {self.le.inverse_transform([klass])[0]: f1 for klass, f1 in enumerate(recall_score(self.test_y, hy, average=None))}
        conf['_all_precision'] = {self.le.inverse_transform([klass])[0]: f1 for klass, f1 in enumerate(precision_score(self.test_y, hy, average=None))}

        if len(self.le.classes_) == 2:
            conf['_macrof1'] = np.mean(np.array([v for v in conf['_all_f1'].values()]))
            conf['_weightedf1'] = conf['_microf1'] = f1_score(self.test_y, hy, average='binary')
        else:
            conf['_macrof1'] = f1_score(self.test_y, hy, average='macro')
            conf['_microf1'] = f1_score(self.test_y, hy, average='micro')
            conf['_weightedf1'] = f1_score(self.test_y, hy, average='weighted')

        conf['_accuracy'] = accuracy_score(self.test_y, hy)
        if self.score.startswith('avgf1:'):
            _, k1, k2 = self.score.split(':')
            conf['_' + self.score] = (M[k1] + M[k2]) / 2

        conf['_score'] = conf['_' + self.score]


class ScoreKFoldWrapper(ScoreSampleWrapper):
    def __init__(self, X, y, nfolds=5, score='macrof1', classifier=ClassifierWrapper, random_state=None):
        self.nfolds = nfolds
        self.score = score
        self.X = X
        self.le = preprocessing.LabelEncoder().fit(y)
        self.y = np.array(self.le.transform(y))
        self.test_y = self.y
        self.create_classifier = classifier
        self.kfolds = cross_validation.StratifiedKFold(y, n_folds=nfolds, shuffle=True, random_state=random_state)

    def __call__(self, conf_code):
        conf, code = conf_code
        st = time()
        predY = np.zeros(len(self.y))
        for train, test in self.kfolds:
            textmodel = TextModel([self.X[i] for i in train], **conf)
            trainX = [textmodel[self.X[i]] for i in train]
            trainY = [self.y[i] for i in train]
            c = self.create_classifier()
            c.fit(trainX, trainY)
            testX = [textmodel[self.X[i]] for i in test]
            predY[test] = c.predict(testX)

        self.compute_score(conf, predY)
        conf['_time'] = (time() - st) / self.nfolds
        return conf
