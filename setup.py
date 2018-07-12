# Copyright 2016-2018 Eric S. Tellez <eric.tellez@infotec.mx>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup
import microtc


long_desc = ''

setup(
    name="microtc",
    description="""A generic minimal text classifier""",
    long_description=long_desc,
    version=microtc.__version__,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],

    packages=['microtc', 'microtc/tests', 'microtc/tools'],
    include_package_data=True,
    zip_safe=False,
    package_data={
        'microtc/tests': ['text.json'],
        'microtc/resources': ['emoticons.json',
                              "english.neg.stopwords",
                              "english.stopwords",
                              "italian.neg.stopwords",
                              "italian.stopwords",
                              "spanish.neg.stopwords",
                              "spanish.stopwords"],
    },
    scripts=[
        'microtc/tools/microtc-train',
        'microtc/tools/microtc-retrain',
        'microtc/tools/microtc-predict',
        'microtc/tools/microtc-params',
        'microtc/tools/microtc-textModel',
        'microtc/tools/microtc-voc',
        'microtc/tools/microtc-perf',
        'microtc/tools/microtc-ensemble',
        'microtc/tools/microtc-kfolds'
    ]
)
