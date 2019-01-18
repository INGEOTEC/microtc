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
import numpy as np
import sys


def test_nparams():
    from microtc.command_line import params
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    params('-k', '2', '-s', '11', '-o', fname + ".tmp", fname)
    os.unlink(fname + ".tmp")


def test_main():
    from microtc.command_line import params
    import os
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', fname)
    os.unlink(output)


def test_pool():
    from microtc.command_line import params
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    params('-k', '2', '-s', '11', '-n', '2', '-o', fname + ".params", fname)
    os.unlink(fname + ".params")


def test_output():
    from microtc.command_line import params
    import os
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', fname)
    assert os.path.isfile(output)
    os.unlink(output)


def test_seed():
    try:
        from mock import MagicMock
    except ImportError:
        from unittest.mock import MagicMock
    from microtc.command_line import params
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    seed = np.random.seed
    np.random.seed = MagicMock()
    params('-s', '2', '--seed', '1', '-k', '2', '-o', fname + ".params", fname)
    os.unlink(fname + ".params")
    np.random.seed.assert_called_once_with(1)
    np.random.seed = seed


def test_train():
    from microtc.command_line import params, train
    import os
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv[0] = 'test-train:microtc-params'
    params('-o', output, '-k', '2', fname, '-s', '2')
    assert os.path.isfile(output)
    with open(output) as fpt:
        print(fpt.read())

    sys.argv[0] = 'test-train:microtc-train'
    print(train('-m', output, '-o', output + ".model", fname))
    os.unlink(output)
    os.unlink(output + '.model')


def test_train_regression():
    from microtc.command_line import params, train, predict
    import os
    import sys
    import tempfile

    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', '-s', '2', '-S', 'r2', fname)
    with open(output) as fpt:
        print("XXXXX>", output, fpt.read())

    train('-o', output + '.model', '-R', '-m', output, fname)
    for x in predict('-m', output + '.model', fname, '-o', output + '.predicted'):
        print(x, file=sys.stderr)

    os.unlink(output)
    os.unlink(output + '.model')
    os.unlink(output + '.predicted')


def test_train2():
    from microtc.command_line import train, params
    import os
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', fname, '-s', '2')
    assert os.path.isfile(output)
    output2 = tempfile.mktemp()
    train('-m', output, fname, '-o', output2)
    os.unlink(output)
    os.unlink(output2)


def test_test():
    from microtc.command_line import params, train, predict
    from microtc.utils import read_data_labels
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '0.5:0.5', fname, '-s', '2')
    params('-o', output, '-k', '2', fname, '-s', '2')
    train('-o', output + '.model', '-m', output, fname)
    predict('-m', output + '.model', fname, '-o', output + '.predicted')
    X, y = read_data_labels(fname)
    print(y)
    os.unlink(output)
    os.unlink(output + '.model')
    os.unlink(output + '.predicted')
    assert len(y)


def test_score():
    from microtc.command_line import params
    import os
    import sys
    import tempfile
    import json
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', '-s', '2', '-S', 'avgf1:POS:NEG', fname)
    with open(output) as fpt:
        a = json.loads(fpt.read())[0]
    assert a['_score'] == a['_avgf1:POS:NEG']
    os.unlink(output)
        

def test_textmodel():
    from microtc.command_line import params, train, textmodel
    import os
    import sys
    import json
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', fname, '-s', '2')
    train('-m', output, fname, '-o', output)
    output2 = tempfile.mktemp()
    textmodel('-m', output, fname, '-o', output2)
    os.unlink(output)
    a = open(output2).readline()
    os.unlink(output2)
    a = json.loads(a)
    assert 'klass' in a


def test_numeric_klass():
    from microtc.utils import tweet_iterator
    from microtc.params import DefaultParams, Fixed
    from microtc.command_line import params, train, predict
    from sklearn.preprocessing import LabelEncoder
    import os
    import json
    import tempfile
    import sys
    numeric = tempfile.mktemp() + '.json'
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    D = [x for x in tweet_iterator(fname)]
    encoder = LabelEncoder().fit([x['klass'] for x in D])
    y = encoder.transform([x['klass'] for x in D])
    for x, k in zip(D, y):
        x['klass'] = int(k)

    with open(numeric, 'w') as fpt:
        [fpt.write(json.dumps(x) + '\n') for x in D]

    P = DefaultParams.copy()
    params('-o', output, '-k', '2', numeric, '-s', '2', **P)
    train('-m', output, numeric, '-o', output)
    output2 = tempfile.mktemp()
    predict('-m', output, fname, '-o', output2)
    os.unlink(numeric)
    os.unlink(output)
    os.unlink(output2)


def test_kfolds():
    from microtc.command_line import params, kfolds
    import os
    import sys
    import json
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    params('-o', output, '-k', '2', fname, '-s', '2')
    output2 = tempfile.mktemp()
    kfolds('-m', output, fname, '-o', output2)
    os.unlink(output)
    a = open(output2).readline()
    os.unlink(output2)
    a = json.loads(a)
    assert 'decision_function' in a
    try:
        kfolds('--update-klass', '-m', output, fname, '-o', output2)
    except AssertionError:
        return
    assert False
