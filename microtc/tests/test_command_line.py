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


def test_nparams():
    from microtc.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['microtc', '-k', '2', '-s', '11', fname]
    c.main()
    os.unlink(c.get_output())


def test_main():
    from microtc.command_line import params
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['microtc', '-o', output, '-k', '2', fname]
    params()
    os.unlink(output)


def test_pool():
    from microtc.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['microtc', '-k', '2', '-s', '11', '-n', '2', fname]
    c.main()
    os.unlink(c.get_output())


def test_output():
    from microtc.command_line import CommandLine
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['microtc', '-o', output, '-k', '2', fname]
    c.main()
    assert os.path.isfile(output)
    os.unlink(output)


def test_seed():
    try:
        from mock import MagicMock
    except ImportError:
        from unittest.mock import MagicMock
    from microtc.command_line import CommandLine
    import os
    import sys
    fname = os.path.dirname(__file__) + '/text.json'
    seed = np.random.seed
    np.random.seed = MagicMock()
    c = CommandLine()
    sys.argv = ['microtc', '-s', '2', '--seed', '1', '-k', '2', fname]
    c.main()
    os.unlink(c.get_output())
    np.random.seed.assert_called_once_with(1)
    np.random.seed = seed


def test_train():
    from microtc.command_line import CommandLine, CommandLineTrain
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['microtc', '-o', output, '-k', '2', fname, '-s', '2']
    c.main()
    assert os.path.isfile(output)
    with open(output) as fpt:
        print(fpt.read())
    c = CommandLineTrain()
    sys.argv = ['microtc', '-m', output, fname]
    print(c.main())
    os.unlink(output)
    os.unlink(c.get_output())
        

def test_train2():
    from microtc.command_line import CommandLine, train
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    c = CommandLine()
    sys.argv = ['microtc', '-o', output, '-k', '2', fname, '-s', '2']
    c.main()
    assert os.path.isfile(output)
    output2 = tempfile.mktemp()
    sys.argv = ['microtc', '-m', output, fname, '-o', output2]
    train()
    os.unlink(output)
    os.unlink(output2)


def test_test():
    from microtc.command_line import params, train, test
    from microtc.utils import read_data_labels
    import os
    import sys
    import tempfile
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['microtc', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['microtc', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp()
    sys.argv = ['microtc', '-m', output, fname, '-o', output2]
    test()
    X, y = read_data_labels(output2)
    print(y)
    os.unlink(output)
    os.unlink(output2)
    assert len(y)


def test_score():
    from microtc.command_line import params
    import os
    import sys
    import tempfile
    import json
    output = tempfile.mktemp()
    fname = os.path.dirname(__file__) + '/text.json'
    sys.argv = ['microtc', '-o', output, '-k', '2', fname, '-s', '2', '-S', 'avgf1:POS:NEG']
    params()
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
    sys.argv = ['microtc', '-o', output, '-k', '2', fname, '-s', '2']
    params()
    sys.argv = ['microtc', '-m', output, fname, '-o', output]
    train()
    output2 = tempfile.mktemp()
    sys.argv = ['microtc', '-m', output, fname, '-o', output2]
    textmodel()
    os.unlink(output)
    a = open(output2).readline()
    os.unlink(output2)
    a = json.loads(a)
    assert 'klass' in a
