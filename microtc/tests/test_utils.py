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