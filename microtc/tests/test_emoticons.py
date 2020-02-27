# Copyright 2020 Mario Graff

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from microtc import emoticons


def download(fname):
    from urllib import request
    import os
    # fname = "https://www.unicode.org/Public/emoji/12.1/emoji-data.txt"
    output = fname.split("/")[-1]
    if os.path.isfile(output):
        return output
    request.urlretrieve(fname, output)
    return output


def test_read_emoji_standard():
    # fname = "/home/mgraffg/software/microtc/microtc/resources/emoji-data.txt"
    data = "https://www.unicode.org/Public/emoji/12.1/emoji-data.txt"
    sec = "https://www.unicode.org/Public/emoji/12.1/emoji-sequences.txt"
    var ="https://www.unicode.org/Public/emoji/12.1/emoji-variation-sequences.txt"
    zwj = "https://www.unicode.org/Public/emoji/12.1/emoji-zwj-sequences.txt"
    emos = emoticons.read_emoji_standard(download(data))
    print(len(emos))
    emoticons.read_emoji_standard(download(sec), emos)
    print(len(emos))
    emoticons.read_emoji_standard(download(var), emos)
    print(len(emos))
    emoticons.read_emoji_standard(download(zwj), emos)
    print(chr(0x1F3FB) in emos, len(emos))
    assert (chr(0x1F468) + chr(0x200D) + chr(0x1F467)) in emos


def test_convert_emoji():
    emo = emoticons.convert_emoji("26BD")
    assert isinstance(emo, str)
    assert chr(0x26bd) == emo
    emos = emoticons.convert_emoji("2753..2755")
    assert isinstance(emos, list)
    assert len(emos) == 3
    print(emos)
    emo = emoticons.convert_emoji("1F4EA FE0E")
    print(emo)
    assert len(emo) == 2 and isinstance(emo, str)


def test_create_data_structure():
    data = "https://www.unicode.org/Public/emoji/12.1/emoji-data.txt"
    zwj = "https://www.unicode.org/Public/emoji/12.1/emoji-zwj-sequences.txt"
    emojis = emoticons.read_emoji_standard(download(data))
    emoticons.read_emoji_standard(download(zwj), emojis)
    data = emoticons.create_data_structure(dict(mario=1, ma=1, g=1, marx=1))
    print(data)
    emoticons.create_data_structure(emojis)


def test_has_emoji():
    data = emoticons.create_data_structure(dict(mario=1, ma=1, g=1, marx=1))
    text = "maxmarxgmlr"
    blocks = emoticons.find_emoji(data, text)
    lst = [text[init:end] for init, end in blocks]
    for a, b in zip(lst, ["ma", "marx", "g"]):
        assert a == b
    print(lst)
    data = "https://www.unicode.org/Public/emoji/12.1/emoji-data.txt"
    zwj = "https://www.unicode.org/Public/emoji/12.1/emoji-zwj-sequences.txt"
    emojis = emoticons.read_emoji_standard(download(data))
    emoticons.read_emoji_standard(download(zwj), emojis)
    data = emoticons.create_data_structure(emojis)
    _ = emoticons.find_emoji(data, chr(0x1F468) + chr(0x200D) + chr(0x1F468))
    print(_)
    init, end = _[0]
    assert (end - init) == 1
    assert chr(0x1F468) in emojis
    assert len(_) == 1
