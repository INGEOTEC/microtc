# author: Eric S. Tellez


def test_params():
    from microtc.params import ParameterSelection
    import numpy as np
    from numpy.random import random
    sel = ParameterSelection()
    
    def fake_score(conf_code):
        conf = conf_code[0]
        conf['_score'] = random()
        conf['_time'] = 1.0
        return conf
        
    sel.search(fake_score, bsize=64)


def test_read_data_labels():
    import os
    from microtc.utils import read_data_labels
    filename = os.path.join(os.path.dirname(__file__), "text.json")
    read_data_labels(filename)


def test_wrapper_score():
    from microtc.scorewrapper import ScoreKFoldWrapper
    from sklearn.metrics import f1_score
    import numpy as np
    np.random.seed(0)
    y = np.random.randint(3, size=100).astype(np.str)
    hy = np.random.randint(3, size=100)
    w = ScoreKFoldWrapper([], y, score='avgf1:0:2', nfolds=10)
    conf = {}
    w.compute_score(conf, hy)
    f1 = f1_score(y.astype(np.int), hy, average=None)
    assert conf['_accuracy'] == (y.astype(np.int) == hy).mean()
    print(y)
    print(conf['_avgf1:0:2'], (f1[0] + f1[2]) / 2.)
    assert conf['_avgf1:0:2'] == (f1[0] + f1[2]) / 2.


def test_counter():
    from microtc.utils import Counter, save_model, load_model
    import os
    c = Counter()
    c.update([1, 2, 3, 1])
    c.update([3])
    assert c[1] == 2
    print(c.update_calls)
    assert c.update_calls == 2
    save_model(c, "t.voc")
    cc = load_model("t.voc")
    os.unlink("t.voc")
    print(cc.update_calls, "**")
    assert cc.update_calls ==  2


def test_counter_add():
    from microtc.utils import Counter

    c1 = Counter()
    c1.update(range(10))
    c2 = Counter()
    c2.update(range(5, 15))
    r = c1 + c2
    print(r)
    assert isinstance(r, Counter)
    for i in range(5):
        assert r[i] == 1
    for i in range(5, 10):
        assert r[i] == 2
    for i in range(10, 15):
        assert r[i] == 1 
    assert r.update_calls == 2


def test_counter_sub():
    from microtc.utils import Counter

    c1 = Counter()
    c1.update(range(10))
    c2 = Counter()
    c2.update(range(5, 15))
    r = c1 + c2
    re = r - c1
    print(re)
    assert isinstance(re, Counter)
    for k, v in re.items():
        assert c2[k] == v
    for k, v in c2.items():
        assert re[k] == v
    assert re.update_calls == 1


def test_counter_json():
    from microtc.utils import Counter
    c1 = Counter()
    c1.update(list(map(str, range(10))))
    print(c1)
    js = c1.tojson()
    print(js)
    c2 = Counter.fromjson(js)
    assert c1.update_calls == c2.update_calls
    print(c2)
    for x, v in c1.items():
        print(x, v, c2[x])
        assert x in c2 and v == c2[x]


def test_sparse_matrix():
    from microtc.utils import SparseMatrix

    class SM(SparseMatrix):
        @property
        def num_terms(self):
            return 4

    matrix = [[(1, 0.5), (3, -0.2)], [(2, 0.3)], [(0, 1), (3, -1.2)]]
    r = SM().tonp(matrix)
    assert r.toarray().shape[1] == 4