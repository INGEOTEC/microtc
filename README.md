[![Build Status](https://travis-ci.org/INGEOTEC/microtc.svg?branch=master)](https://travis-ci.org/INGEOTEC/microtc)
[![Build Status](https://travis-ci.org/INGEOTEC/microtc.svg?branch=develop)](https://travis-ci.org/INGEOTEC/microtc)
[![Build status](https://ci.appveyor.com/api/projects/status/afcwh0d9sw6g937h?svg=true)](https://ci.appveyor.com/project/mgraffg/microtc)
[![Build status](https://ci.appveyor.com/api/projects/status/afcwh0d9sw6g937h/branch/master?svg=true)](https://ci.appveyor.com/project/mgraffg/microtc/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/microTC/badge.svg?branch=master)](https://coveralls.io/github/INGEOTEC/microTC?branch=master)
[![Anaconda-Server Badge](https://anaconda.org/INGEOTEC/microtc/badges/version.svg)](https://anaconda.org/INGEOTEC/microtc)
[![Anaconda-Server Badge](https://anaconda.org/INGEOTEC/microtc/badges/latest_release_date.svg)](https://anaconda.org/INGEOTEC/microtc)
[![Anaconda-Server Badge](https://anaconda.org/INGEOTEC/microtc/badges/platforms.svg)](https://anaconda.org/INGEOTEC/microtc)
[![Anaconda-Server Badge](https://anaconda.org/INGEOTEC/microtc/badges/installer/conda.svg)](https://conda.anaconda.org/INGEOTEC)
[![PyPI version](https://badge.fury.io/py/microtc.svg)](https://badge.fury.io/py/microtc)
[![Anaconda-Server Badge](https://anaconda.org/INGEOTEC/microtc/badges/license.svg)](https://anaconda.org/INGEOTEC/microtc)
# What is microTC? #

MicroTC follows a minimalist approach to text classification. It is designed to tackle text-classification problems in an agnostic way,
being both domain and language independent.  Currently, we only produce single-label classifiers; but support for multi-labeled problems is in the roadmap.

$\mu$TC is intentionally simple, so only a small number of features where implemented. However, it uses a some complex tools from numpy and scikit-lean. The number of dependencies is limited and fullfilled by almost any Scientific Python distributions, e.g., [Anaconda](https://www.continuum.io/downloads).

## Citing ##

If you find $\mu$TC useful for any academic/scientific purpose, we would appreciate citations to the following reference:

[An Automated Text Categorization Framework based on Hyperparameter Optimization](https://www.sciencedirect.com/science/article/pii/S0950705118301217)
Eric S. Tellez, Daniela Moctezuma, Sabino Miranda-Jímenez, Mario Graff. Knowledge-Based Systems
Volume 149, 1 June 2018, Pages 110-123


```bibtex

@article{Tellez2018110,
title = "An automated text categorization framework based on hyperparameter optimization",
journal = "Knowledge-Based Systems",
volume = "149",
pages = "110--123",
year = "2018",
issn = "0950-7051",
doi = "10.1016/j.knosys.2018.03.003",
url = "https://www.sciencedirect.com/science/article/pii/S0950705118301217",
author = "Eric S. Tellez and Daniela Moctezuma and Sabino Miranda-Jiménez and Mario Graff",
keywords = "Text classification",
keywords = "Hyperparameter optimization",
keywords = "Text modelling"
}
```

# Installing $\mu$TC
MicroTC can be easily installed, in almost scientific python distribution
```bash

git clone https://github.com/INGEOTEC/microTC.git
cd microTC
python setup.py install --user
```

It can be installed from the scratch (Linux system) using the following code (excerpt from [travis-cl.org](https://travis-ci.org/INGEOTEC/microTC))
```bash

$ wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
$ chmod 755 miniconda.sh
$ ./miniconda.sh -b
$ export PATH=$HOME/miniconda3/bin:$PATH
$ pip install numpy
$ pip install scipy
$ pip install scikit-learn
$ pip install nose
$ git clone https://github.com/INGEOTEC/microTC.git
$ cd microTC
$ nosetests
$ python setup.py install --user
```
It supposes you have git installed in your system. If you don't have it, you can install it using `apt-get`, `yum`, etc., or simply downloading a the latest version directly from the [repository](https://github.com/INGEOTEC/microTC).

# Using $\mu$TC
For any given text classification task, $\mu$TC will try to find the best text model from all possible models as defined in the configuration space (please check 
```bash
microTC-params -k3  -Smacrorecall -s24 -n24 user-profiling.json -o user-profiling.params
```

the parameters means for:

- `user-profiling.json` is database of exemplars, one json-dictionary per line with text and klass keywords
- `-k3` three folds
- `-s24` specifies that the parameter space should be sampled in 24 points and then get the best among them
- `-n24` let us specify the number of processes to be launch, it is a good idea to set `-s` as a multiply of `-n`.
- `-o user-profiling.params` specifies the file to store the configurations found by the parameter selection process, in best first order
- `-S` or `--score` the name of the fitness function (e.g., macrof1, microf1, macrorecall, accuracy, r2, pearsonr, spearmanr)
- `-H` makes b4msa to perform last hill climbing search for the parameter selection, in many cases, this will produce much better configurations (never worst, guaranteed)
- all of these parameters have default values, such as no arguments are needed

Notes:
- "text"  can be a string or an array of strings, in the last case, the final vector considers each string as independent strings.
- there is no typo, we use "klass" instead of "class" because of oscure historical reasons
- `-k` accepts an a:b syntax that allow searching in a sample of size $a$ and test in $b$; for $0 < a < 1$, and $0 < b < 1$. It is common to use $b = 1 - a$; however, it is not a hard constraint and just you need to follow `a + b <= 1` and no overlapping ranges.
- If `-S` is `r2`, `pearsonr`, or `spearmanr` then microTC computes the parameters for a regression task.


TODO: Explain environment variables TOKENLIST, PARAMS
# Input format:
The input dataset is quite simple:
- each line is an example
- each example is a valid json dictionary, with two special keywords _"text"_ and _"klass"_
  - _"text"_ is the object's content
  - _"klass"_ is the label of the object
  - you can specify other keywords with the TEXT and KLASS environment variables

# Training the model 

At this point, we are in the position to train a model.
Let us that the workload is `emotions.json` and that the parameters are in
`emotions.params` then the following command will save the model in `emotions.model`

```bash
microtc-train -o emptions.model -m emotions.params emotions.json
```

You can create a regressor adding the `-R` option to `microtc-train`

# Testing the classifier against a workload

At this point, we are in the position to test the model (i.e,
`emotions.model`) in a new set. That is, we are in the position to ask
the classifier to assign a label to a particular text.

```bash
microtc-predict -m emotions.model -o emotions-predicted.json test-emotions.json
```

Finally, you can evaluate the performance of the prediction as follows:

```bash
microtc-perf gold.json emotions-predicted.json
```
This will show a number of scores in the screen.

```json
{
    "accuracy": 0.7025,
    "f1_anger": 0.705,
    "f1_fear": 0.6338797814207651,
    "f1_joy": 0.7920353982300885,
    "f1_sadness": 0.6596858638743456,
    "macrof1": 0.6976502608812997,
    "macrof1accuracy": 0.490099308269113,
    "macrorecall": 0.7024999999999999,
    "microf1": 0.7025,
    "quadratic_weighted_kappa": 0.5773930753564155
}
```
or, in case of provide the `--regression` flag
```json
{
    "filename": "some-path/some-name.predicted",
    "pearsonr": [
        0.6311471948385253,
        1.2734619266038659e-23
    ],
    "r2": 0.3276512897198096,
    "spearmanr": [
        0.6377984613587965,
        3.112636137077516e-24
    ]
}
```

# Minimum requirements
In the modeling stage, the minimum requirements are dependent on the knowledge database being processed. Make sure you have enough memory for it. Take into account that microTC can take advantage of multicore architectures using the `multiprocessing` module of python, this means that the memory requirements are multiplied by the number of processes you run.

It is recomended to use as many cores as you have to obtain good results in short running times.

On the training and testing stages only one core is used and there is no extra memory needs; however, no multicore support is provided for these stages.

# Installing dependencies

Let us download python (from conda distribution), install it, and include python
in the PATH.

```bash
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod 755 miniconda.sh
./miniconda.sh -b
export PATH=/home/$USER/miniconda3/bin:$PATH
```

B4MSA needs the following dependencies.

```bash
pip install coverage
pip install numpy
pip install scipy
pip install scikit-learn
pip install nose
pip install nltk
```

For the eager people, it is recommended to install the `tqdm` package

```bash
pip install tqdm
```

# Where I can find some datasets? ##

- Ana Cachopo's provides a number of [datasets](http://ana.cachopo.org/datasets-for-single-label-text-categorization),
    some of them are already preprocessed.
- [Moschitti's](http://disi.unitn.it/moschitti/corpora.htm) dataset list.
- An unprocessed version of [20-news](http://qwone.com/~jason/20Newsgroups/)

## Multilingual sentiment analysis ###
- English [SemEval 2016](http://alt.qcri.org/semeval2016/)
- Spanish [TASS 2016](http://www.sepln.org/workshops/tass/2016/tass2016.php)
- Italian [SENTIPOLC 2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/)

A list of datasets for 14 number of languages is provided by [MozetiÄ I et al. 2016](http://dx.doi.org/10.1371/journal.pone.0155036).

We provide a reduced version of four languages with the intention of preserve a number of tweets over the time, i.e, some tweets are unaccessible over the time.  We also preserve the class' proportions of the original datasets, in order to capture the spirit of the original dataset. Finally, these datasets are prepared in the microTC's format.

- [German](TBD/german). 
- [Swedish](TBD/swedish).
- [Portuguese](TBD/portuguese)
- [Russian](TBD/russian).


