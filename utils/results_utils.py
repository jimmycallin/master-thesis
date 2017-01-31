import pandas as pd
import os
from os.path import join
import sys
import json
from collections import defaultdict

class Prototext():
    def __init__(self, prototext_filepath):
        self.prototext_filepath = prototext_filepath
        with open(prototext_filepath) as f:
            self._parse_prototext(f.readlines())
    def _parse_prototext(self, lines):
        self.prototext_dict = {}
        self.conf_matrix = {}
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith("measure"):
                continue
            elif line.strip().startswith("key:"):
                key = line.strip()[6:-1]
                line = next(lines_iter)
                value = float(line.strip()[8:-1])
                self.prototext_dict[key] = value
            elif "confusion matrix" in line:
                res_type = line[:line.index("confusion matrix")].strip()
                self.conf_matrix[res_type] = defaultdict(dict)
                line = next(lines_iter)
                header_dict = {i: x.strip() for i, x in enumerate(line.split(","))}

                while line.strip() != "}":
                    line = next(lines_iter)
                    split = line.strip().split(",")
                    for k, val in enumerate(split[1:]):
                        self.conf_matrix[res_type][split[0]][header_dict[k]] = float(val)

    @property
    def non_explicit_f1(self):
        return self.prototext_dict["Non-explicit only Parser f1"]

    @property
    def non_explicit_precision(self):
        return self.prototext_dict["Non-explicit only Parser precision"]

    @property
    def non_explicit_recall(self):
        return self.prototext_dict["Non-explicit only Parser recall"]

    @property
    def non_explicit_conf_matrix(self):
        return pd.DataFrame.from_dict(self.conf_matrix['Non-explicit only'], orient='index')

    def __getitem__(self, val):
        return self.prototext_dict[val]


def proto2pandas(proto_objects, sense=None):
    res_dict = {}
    for name, proto in proto_objects.items():
        if name.startswith("cnn"):
            continue
        if sense is None:
            res_dict[name] = proto.non_explicit_f1
        else:
            res_dict[name] = proto[sense]
    return pd.Series(res_dict)

def proto2confs_grouped(proto_objects):
    res_dict = defaultdict(dict)
    for name, proto in proto_objects.items():
        if name.startswith("cnn"):
            continue
        architecture = name.split("-")[0]
        embedding_type = name[len(architecture)+1:name.index('conll1')-1]
        res_dict[architecture][embedding_type] = proto.non_explicit_conf_matrix
    return res_dict

def get_senses():
    files = {"blind-test": "conll15st-en-03-29-16-blind-test", "dev": "conll16st-en-03-29-16-dev", "test": "conll16st-en-03-29-16-test", "trial": "conll16st-en-03-29-16-trial"}

    files = {name: "../resources/conll16st-en-zh-dev-train-test_LDC2016E50/" + f + "/relations.json" for name, f in files.items()}

    sense_dict = dict()
    for name, f in files.items():
        sense_set = set()
        with open(f) as fi:
            for line in fi:
                js = json.loads(line)
                senses = js['Sense']
                if js['Type'] != 'Explicit':
                    for sense in senses:
                        sense_set.add(sense)
        sense_dict[name] = sense_set
    print("Extracted senses: " + str(sense_dict))
    return sense_dict

def index2tuple(s):
    t = (s[:s.index("-")], s[s.index("-")+1:])
    t = (t[0], t[1][:t[1].index('-conll1')])
    return t

def series2matrix(series):
    multiindex = pd.MultiIndex.from_tuples([index2tuple(k) for k, v in series.iteritems()])
    series.index = multiindex
    return series.unstack(level=0)


def get_confusion_matrixes(test_type):
    test_types = {'dev': 'dev', 'test': '6-test', 'trial': 'trial', 'blind-test': 'blind-test'}

    proto_files = [x for x in os.listdir('../results') if x.endswith('.prototext')]
    proto_objects = {f[:-10]: Prototext("../results/{}".format(f)) for f in proto_files}
    results = set(filter(lambda x: x.endswith(test_types[test_type]), proto_objects))
    res_series = proto2confs_grouped({k: v for k,v in proto_objects.items() if k in results})
    return res_series

def generate_matrix(test_type, sense=None):
    test_types = {'dev': 'dev', 'test': '6-test', 'trial': 'trial', 'blind-test': 'blind-test'}

    proto_files = [x for x in os.listdir('../results') if x.endswith('.prototext')]
    proto_objects = {f[:-10]: Prototext("../results/{}".format(f)) for f in proto_files}
    results = set(filter(lambda x: x.endswith(test_types[test_type]), proto_objects))
    if sense is None:
        res_series = proto2pandas({k: v for k,v in proto_objects.items() if k in results})
    else:
        res_series = proto2pandas({k: v for k,v in proto_objects.items() if k in results}, sense='Non-explicit ' + sense + ' F1')
    df = series2matrix(res_series)
    df = df.append(pd.DataFrame({'average': df.sum(axis=0) / df.count(axis=0), 'variance': df.var(axis=0)}).T)
    df['average'] = df.sum(axis=1) / df.count(axis=1)
    df['variance'] = df.var(axis=1)

    return df

def generate_latex_matrix(test_type, output_file, sense=None):
    df_matrix = generate_matrix(test_type, sense)
    with open(output_file, 'w') as f:
        f.write(df_matrix.to_latex())

def generate_all_latex_matrices(output_folder):
    test_types = ['blind-test', 'dev', 'test', 'trial']
    for test_type in test_types:
        senses = [None] + list(get_senses()[test_type])
        for sense in senses:
            if sense is None:
                output_file_name = "result_matrix_overall_" + test_type + '.table'
            else:
                output_file_name = "result_matrix_" + sense + "_" + test_type + ".table"
            generate_latex_matrix(test_type, join(output_folder, output_file_name), sense)
            print("Wrote " + join(output_folder, output_file_name))
