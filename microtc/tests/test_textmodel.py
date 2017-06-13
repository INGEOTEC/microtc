# Copyright 2016 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_tweet_iterator():
    import os
    import gzip
    from microtc.utils import tweet_iterator
    
    fname = os.path.dirname(__file__) + '/text.json'
    a = [x for x in tweet_iterator(fname)]
    fname_gz = fname + '.gz'
    with open(fname, 'r') as fpt:
        with gzip.open(fname_gz, 'w') as fpt2:
            fpt2.write(fpt.read().encode('ascii'))
    b = [x for x in tweet_iterator(fname_gz)]
    assert len(a) == len(b)
    for a0, b0 in zip(a, b):
        assert a0['text'] == b0['text']
    os.unlink(fname_gz)


def test_textmodel():
    from microtc.textmodel import TextModel
    from microtc.utils import tweet_iterator
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    tw = list(tweet_iterator(fname))
    text = TextModel([x['text'] for x in tw])
    # print(text.tokenize("hola amiguitos gracias por venir :) http://hello.com @chanfle"))
    # assert False
    assert isinstance(text[tw[0]['text']], list)


def test_params():
    import os
    import itertools
    from microtc.params import BASIC_OPTIONS
    from microtc.textmodel import TextModel
    from microtc.utils import tweet_iterator

    params = dict(strip_diac=[True, False], usr_option=BASIC_OPTIONS,
                  url_option=BASIC_OPTIONS)
    params = sorted(params.items())
    fname = os.path.dirname(__file__) + '/text.json'
    tw = [x for x in tweet_iterator(fname)]
    text = [x['text'] for x in tw]
    for x in itertools.product(*[x[1] for x in params]):
        args = dict(zip([x[0] for x in params], x))
        ins = TextModel(text, **args)
        assert isinstance(ins[text[0]], list)


def test_lang():
    from microtc.textmodel import TextModel

    text = [
        "Hi :) :P XD",
        "excelente dia xc",
        "el alma de la fiesta XD"
    ]
    model = TextModel(text, **{
        "del_dup1": True,
        "emo_option": "group",
        "lc": True,
        "num_option": "group",
        "strip_diac": False,
        "token_list": [
            (2, 1),
            (2, 2),
            -1,
            # 5,
        ],
        "url_option": "group",
        "usr_option": "group",
    })
    text = "El alma de la fiesta :) conociendo la maquinaria @user bebiendo nunca manches que onda"
    a = model.tokenize(text)
    b = ['el~de', 'alma~la', 'de~fiesta', 'la~_pos', 'fiesta~conociendo', '_pos~la', 'conociendo~maquinaria', 'la~_usr', 'maquinaria~bebiendo', '_usr~nunca',
         'bebiendo~manches', 'nunca~que', 'manches~onda', 'el~la', 'alma~fiesta', 'de~_pos', 'la~conociendo', 'fiesta~la', '_pos~maquinaria', 'conociendo~_usr',
         'la~bebiendo', 'maquinaria~nunca', '_usr~manches', 'bebiendo~que', 'nunca~onda', 'el', 'alma', 'de', 'la', 'fiesta', '_pos',
         'conociendo', 'la', 'maquinaria', '_usr', 'bebiendo', 'nunca', 'manches', 'que', 'onda']
    print(text)
    assert a == b, "got: {0}, expected: {1}".format(a, b)





