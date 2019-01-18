.. EvoMSA documentation master file, created by
   sphinx-quickstart on Fri Aug  3 07:02:12 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:math:`\mu\text{TC}`
==================================

A great variety of text tasks such as topic or spam identification,
user profiling, and sentiment analysis can be posed as a supervised
learning problem and tackled using a text classifier. A text
classifier consists of several subprocesses, some of them are general
enough to be applied to any supervised learning problem, whereas
others are specifically designed to tackle a particular task using
complex and computational expensive processes such as lemmatization,
syntactic analysis, etc. Contrary to traditional approaches,
:math:`\mu\text{TC}` is a minimalist and multi-propose text-classifier able to tackle
tasks independently of domain and language.

:math:`\mu\text{TC}` is described in
`An Automated Text Categorization Framework based on Hyperparameter Optimization <https://www.sciencedirect.com/science/article/pii/S0950705118301217>`_.
Eric S. Tellez, Daniela Moctezuma, Sabino Miranda-Jímenez, Mario Graff. Knowledge-Based Systems
Volume 149, 1 June 2018, Pages 110-123


Citing
======

If you find :math:`\mu\text{TC}` useful for any academic/scientific purpose, we
would appreciate citations to the following reference:

.. code:: bibtex
	
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

Installing :math:`\mu\text{TC}`
===============================

:math:`\mu\text{TC}` can be easly install using anaconda

.. code:: bash

	  conda install -c ingeotec microtc

or can be install using pip, it depends on numpy, scipy and
scikit-learn.

.. code:: bash
	  
	  pip install numpy
	  pip install scipy
	  pip install scikit-learn
	  pip install microtc


Text Model
=============

This is class is :math:`\mu\text{TC}` main entry, it receives a
corpus, i.e., a list of text and builds a text model from it.

.. autoclass:: microtc.textmodel.TextModel
   :members:

Modules
==================	      

.. toctree::
   :maxdepth: 2

   weighting
   utils
