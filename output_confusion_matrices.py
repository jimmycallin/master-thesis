import pandas as pd
from sys import argv
import os

class Prototext():
    def __init__(self, prototext_filepath):
        with open(prototext_filepath) as f:
            self._parse_prototext(f.readlines())
    def _parse_prototext(self, lines):
        self.prototext_dict = {}
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith("measure"):
                continue
            elif line.strip().startswith("key:"):
                key = line.strip()[6:-1]
                line = next(lines_iter)
                value = line.strip()[8:-1]
                self.prototext_dict[key] = value
    @property
    def non_explicit_f1(self):
        return self.prototext_dict["Non-explicit only Parser f1"]

    @property
    def non_explicit_precision(self):
        return self.prototext_dict["Non-explicit only Parser precision"]

    @property
    def non_explicit_recall(self):
        return self.prototext_dict["Non-explicit only Parser recall"]

    def __getitem__(self, val):
        return self.prototext_dict[val]


def proto2pandas(proto_objects):
    res_dict = {}
    for name, proto in proto_objects.items():
        res_dict[name] = proto.non_explicit_f1
    return pd.Series(res_dict)

if __name__ == '__main__':
    prototext_keys = {'Non-explicit only Parser f1': 'F1'}
    proto_files = [x for x in os.listdir('results') if x.endswith('.prototext')]
    proto_objects = {f[:-10]: Prototext("results/{}".format(f)) for f in proto_files}
    blind_tests = set(filter(lambda x: x.endswith("blind-test"), proto_objects))
    devs = set(filter(lambda x: x.endswith("dev"), proto_objects))
    tests = set(filter(lambda x: x.endswith("6-test"), proto_objects))
    trials = set(filter(lambda x: x.endswith("trial"), proto_objects))

    proto2pandas({k: v for k,v in proto_objects.items() if k in devs})
