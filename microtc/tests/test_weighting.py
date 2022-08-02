# Copyright 2018 Mario Graff

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from os.path import join


def test_space():
    from microtc.textmodel import TextModel
    from microtc.weighting import TFIDF
    from microtc.utils import tweet_iterator
    import os
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    assert len(sp.wordWeight) == len(sp._w2id)
    # print(sp._weight)
    # print(sp._w2id)


def test_doc2weight():
    from microtc.textmodel import TextModel
    from microtc.weighting import TFIDF
    from microtc.utils import tweet_iterator
    import os
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    assert len(sp.doc2weight(text.tokenize('odio odio los los'))) == 3


def test_getitem():
    from microtc.textmodel import TextModel
    from microtc.weighting import TFIDF
    from microtc.utils import tweet_iterator
    import os
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(docs, token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = TFIDF(docs)
    tok = text.tokenize('buenos dias')
    bow = sp.doc2weight(tok)
    ids = bow[0]
    assert len(ids) == len(sp[tok])


def test_entropy():
    from microtc.textmodel import TextModel
    from microtc.weighting import Entropy, TFIDF
    from microtc.utils import tweet_iterator
    import os
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(token_list=[-1, 3])
    # print(text['buenos dias'])
    docs = [text.tokenize(d) for d in docs]
    sp = Entropy(docs, X=tw)
    print(sp.wordWeight)
    tfidf = TFIDF(docs)
    for k in sp.wordWeight.keys():
        if sp.wordWeight[k] != tfidf.wordWeight[k]:
            return
    # print(sp.w)
    assert False


def test_tfidf_corpus():
    from nose.tools import assert_almost_equals
    from microtc.textmodel import TextModel
    from microtc.weighting import TFIDF
    from microtc.utils import Counter
    from microtc.utils import tweet_iterator
    import os
    import numpy as np
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    text = TextModel(token_list=[-1, 3])
    docs = [text.tokenize(d) for d in docs]
    counter = Counter()
    [counter.update(set(x))for x in docs]
    tfidf = TFIDF(docs)
    tfidf2 = TFIDF.counter(counter)
    assert tfidf.num_terms == tfidf2.num_terms
    assert tfidf._ndocs == tfidf2._ndocs
    for k in tfidf2.word2id.keys():
        assert k in tfidf2.word2id
    for k, v in tfidf.word2id.items():
        id2 = tfidf2.word2id[k]
        v = tfidf.wordWeight[v]
        v2 = tfidf2.wordWeight[id2]
        print(v, v2, k)
        assert_almost_equals(v, v2)


def test_tfidf_corpus2():
    from nose.tools import assert_almost_equals
    from microtc.textmodel import TextModel
    from microtc.weighting import TFIDF
    from microtc.utils import Counter
    from microtc.utils import tweet_iterator
    import os
    import numpy as np
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    tm = TextModel(token_list=[-1, 3])
    docs = [tm.tokenize(d) for d in docs]
    counter = Counter()
    [counter.update(set(x))for x in docs]
    tfidf = TFIDF(docs, token_min_filter=1)
    tfidf2 = TFIDF.counter(counter, token_min_filter=1)
    id2w2 = {v: k for k, v in tfidf2.word2id.items()}
    for text in docs:
        tokens = tm.tokenize(text)
        fm = {k: v for k, v in tfidf[tokens]}
        for k, v in tfidf2[tokens]:
            assert_almost_equals(fm[tfidf.word2id[id2w2[k]]], v)


def test_max_dimension():
    from microtc import TextModel
    from microtc.utils import tweet_iterator
    import os
    fname = join(os.path.dirname(__file__), 'text.json')
    tw = list(tweet_iterator(fname))
    docs = [x['text'] for x in tw]
    tm = TextModel(token_list=[-1, 2, 3, 4],
                   token_max_filter=2**4,
                   max_dimension=True).fit(docs)
    assert tm.num_terms == 2**4
    tm2 = TextModel(token_list=[-1, 2, 3, 4]).fit(docs)
    assert tm2.num_terms > tm.num_terms
    assert not tm2.max_dimension  