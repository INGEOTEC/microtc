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

__version__ = "1.6"


# class WorkerTC:
#     def __init__(self, filename):
#         with open(filename, 'rb') as f:
#             t, c, le = pickle.load(f)

#         self.model = t
#         self.svc = c
#         self.le = le

#     def predict_dict(self, tweet):
#         tweet['klass'] = self.predict(tweet['text'])
#         return tweet

#     def predict(self, text):
#         vec = self.model[text]
#         return self.le.inverse_transform(self.svc.predict([vec]))[0]
