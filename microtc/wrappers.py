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
import numpy as np
from scipy.sparse import csr_matrix


class ClassifierWrapper(object):
    def __init__(self, algo=LinearSVC):
        # from sklearn.neighbors.nearest_centroid import NearestCentroid
        # self.svc = NearestCentroid(metric='cosine', shrink_threshold=None)
        # from sklearn.neighbors import KNeighborsClassifier
        # self.svc = KNeighborsClassifier(3, metric='cosine')
        self.svc = algo()

    @property
    def num_terms(self):
        """Dimension which is the number of terms of the corpus

        :rtype: int
        """

        try:
            return self._num_terms
        except AttributeError:
            self._num_terms = None
        return None

    def tonp(self, X):
        """Sparse representation to sparce matrix

        :param X: Sparse representation of matrix
        :type X: list
        :rtype: csr_matrix
        """

        data = []
        row = []
        col = []
        for r, x in enumerate(X):
            cc = [_[0] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            col += cc
            data += [_[1] for _ in x if np.isfinite(_[1]) and (self.num_terms is None or _[0] < self.num_terms)]
            _ = [r] * len(cc)
            row += _
        if self.num_terms is None:
            _ = csr_matrix((data, (row, col)))
            self._num_terms = _.shape[1]
            return _
        return csr_matrix((data, (row, col)), shape=(len(X), self.num_terms))

    def fit(self, X, y):
        X = self.tonp(X)
        self.svc.fit(X, y)
        return self

    def decision_function(self, Xnew):
        Xnew = self.tonp(Xnew)
        return self.svc.decision_function(Xnew)

    def predict_proba(self, Xnew):
        Xnew = self.tonp(Xnew)
        return self.svc.predict_proba(Xnew)

    def predict(self, Xnew):
        Xnew = self.tonp(Xnew)
        ynew = self.svc.predict(Xnew)
        return ynew


class RegressorWrapper(ClassifierWrapper):
    def __init__(self, algo=LinearSVR):
        super(RegressorWrapper, self).__init__(algo=algo)
