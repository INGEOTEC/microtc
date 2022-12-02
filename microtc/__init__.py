# Copyright 2016-2018 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""":math:`\mu\text{TC}`
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

The starting point is :py:class:`microtc.textmodel.TextModel`

"""
__version__ = "2.4.4"

from microtc.textmodel import TextModel
