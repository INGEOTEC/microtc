import json
import io

def readlines(filename):
    with io.open(filename, encoding='utf8') as f:
        return f.readlines()


def dump(filename, klass):
    for line in readlines(filename):
        line = line.rstrip()
        d = {'text': line, 'klass': klass}
        print(json.dumps(d))

import sys
dump(sys.argv[1], sys.argv[2])

