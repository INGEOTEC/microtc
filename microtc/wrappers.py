# Copyright 2016 Ranyart R. Suarez (https://github.com/RanyartRodrigo) and Mario Graff (https://github.com/mgraffg)
# with collaborations of Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.svm import LinearSVC, LinearSVR
from gensim.matutils import corpus2csc
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s :%(message)s')


class ClassifierWrapper(object):
    def __init__(self, algo=LinearSVC):
        # from sklearn.neighbors.nearest_centroid import NearestCentroid
        # self.svc = NearestCentroid(metric='cosine', shrink_threshold=None)
        # from sklearn.neighbors import KNeighborsClassifier
        # self.svc = KNeighborsClassifier(3, metric='cosine')
        self.svc = algo()
        self.num_terms = -1

    def fit(self, X, y):
        X = corpus2csc(X).T
        self.num_terms = X.shape[1]
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        return self.svc.decision_function(Xnew)

    def predict_proba(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        return self.svc.predict_proba(Xnew)

    def predict(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        ynew = self.svc.predict(Xnew)
        return ynew


class RegressorWrapper(object):
    def __init__(self, algo=LinearSVR):
        self.svc = algo()
        self.num_terms = -1

    def fit(self, X, y):
        X = corpus2csc(X).T
        self.num_terms = X.shape[1]
        self.svc.fit(X, y)
        return self

    def predict(self, Xnew):
        Xnew = corpus2csc(Xnew, num_terms=self.num_terms).T
        return self.svc.predict(Xnew)
