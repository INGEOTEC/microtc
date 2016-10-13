[![Build Status](https://travis-ci.org/INGEOTEC/microTC.svg?branch=master)](https://travis-ci.org/INGEOTEC/microTC)

[![Coverage Status](https://coveralls.io/repos/github/INGEOTEC/microTC/badge.svg?branch=master)](https://coveralls.io/github/microTC/microtc?branch=master)

# What is microTC? #

microTC follows a minimalistic approach to text classification. It is designed to tackle text-classification problems in an agnostic way,
being both domain and language independent.  Currently, we only produce single-label classifiers; but support for multi-labeled problems is in the roadmap.

$micro$TC is intentionally simple, so only a small number of features where implemented. However, it uses a some complex tools from gensim, numpy and scikit-lean. The number of dependencies is limited in a Scientific Python distribution like [Anaconda](https://www.continuum.io/downloads).


## Where I can find some datasets? ##

- Ana Cachopo's provides a number of [datasets](http://ana.cachopo.org/datasets-for-single-label-text-categorization),
    some of them are already preprocessed.
- [Moschitti's](http://disi.unitn.it/moschitti/corpora.htm) dataset list.
- An unprocessed version of [20-news](http://qwone.com/~jason/20Newsgroups/)

### Multilingual sentiment analysis ###
- English [SemEval 2016](http://alt.qcri.org/semeval2016/)
- Spanish [TASS 2016](http://www.sepln.org/workshops/tass/2016/tass2016.php)
- Italian [SENTIPOLC 2016](http://www.di.unito.it/~tutreeb/sentipolc-evalita16/)

A list of datasets for 14 number of languages is provided by [Mozetič I et al. 2016](http://dx.doi.org/10.1371/journal.pone.0155036).

We provide a reduced version of four languages with the intention of preserve a number of tweets over the time, i.e, some tweets are unaccessible over the time.  We also preserve the class' proportions of the original datasets, in order to capture the spirit of the original dataset. Finally, these datasets are prepared in the microTC's format.

- [German](TBD/german). 
- [Swedish](TBD/swedish).
- [Portuguese](TBD/portuguese)
- [Russian](TBD/russian).

