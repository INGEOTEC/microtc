# Copyright 2016 Mario Graff (https://github.com/mgraffg) and Ranyart R. Suarez (https://github.com/RanyartRodrigo)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_predict_from_file():
    from microtc.wrappers import ClassifierWrapper
    from microtc.textmodel import TextModel
    from microtc.utils import read_data_labels
    from sklearn.preprocessing import LabelEncoder

    import os
    fname = os.path.dirname(__file__) + '/text.json'
    corpus, labels = read_data_labels(fname)
    t = TextModel(corpus)
    le = LabelEncoder()
    le.fit(labels)
    y = le.transform(labels)
    c = ClassifierWrapper()
    X = [t[x] for x in corpus]
    c.fit(X, y)
    hy = le.inverse_transform(c.predict(X))
    for i in hy:
        assert i in ['POS', 'NEU', 'NEG']

