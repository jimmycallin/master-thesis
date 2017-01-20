#coding:utf-8
import sys
# import imp
# imp.reload(sys)
# sys.setdefaultencoding('utf-8')
sys.path.append("./")
import json
import parser_util
import config
import cnn_test
import pickle

import codecs

def data_process_special(relations_file,parses_file):
    rf = open(relations_file)
    pf = open(parses_file)
    relations = [json.loads(x) for x in rf]
    parse_dict = json.load(codecs.open(parses_file, encoding='utf8'))
    relation = []
    flag= 0
    for r in relations:
        docid = r['DocID']
        type = r['Type']
        sense = r['Sense']
        if type == 'Explicit':
            continue
        ''' arg_offset_list = [sentence_index,wordinsentence_index] from TokenList in relations '''
        arg1_tokenlist = [[t[3],t[4]] for t in r['Arg1']['TokenList']]
        arg2_tokenlist = [[t[3],t[4]] for t in r['Arg2']['TokenList']]
        arg1_word = [parse_dict[docid]["sentences"][sent_index]["words"][word_index][0] for sent_index,word_index in arg1_tokenlist]
        arg1_pos = [parse_dict[docid]["sentences"][sent_index]["words"][word_index][1]["PartOfSpeech"] for sent_index,word_index in arg1_tokenlist]
        arg2_word = [parse_dict[docid]["sentences"][sent_index]["words"][word_index][0] for sent_index,word_index in arg2_tokenlist]
        arg2_pos = [parse_dict[docid]["sentences"][sent_index]["words"][word_index][1]["PartOfSpeech"] for sent_index,word_index in arg2_tokenlist]
        relation.append((arg1_word,arg2_word,arg1_pos,arg2_pos,sense,r))
    return relation

class DiscourseParser():
    def __init__(self, input_dataset):
        self.input_dataset = input_dataset
        self.relations = []
        self.explicit_relations = []
        self.non_explicit_relations = []

    def parse(self):
        dev_file_path = self.input_dataset
        parses_file_name = dev_file_path+"/parses.json"
        relations_file_name = dev_file_path+"/relations.json"
        instances = data_process_special(relations_file_name,parses_file_name)
        cnn_imp_dict = "/Users/jimmy/dev/edu/master-thesis/architectures/CoNLL2016-CNN/model_trainer/cnn_implicit_classifier/4595.txt"
        cnn_imp_model = "/Users/jimmy/dev/edu/master-thesis/architectures/CoNLL2016-CNN/model_trainer/cnn_implicit_classifier/4595.h5"
        cnn_noexp_output = cnn_test.test(instances, cnn_imp_dict, cnn_imp_model)

        self.relations = [i[5] for i in instances]
        from model_trainer.tf_cnn_implicit import cnn_config
        import numpy as np
        for i,r in enumerate(self.relations):
            r['Arg1']['TokenList'] = [t[2] for t in r['Arg1']['TokenList']]
            r['Arg2']['TokenList'] = [t[2] for t in r['Arg2']['TokenList']]
            r["Sense"] = [cnn_config.Label_To_Sense[np.argmax(cnn_noexp_output[i])]]

if __name__ == '__main__':

    # input_dataset = sys.argv[1]
    # input_run = sys.argv[2]
    # output_dir = sys.argv[3]

    input_dataset = sys.argv[2]

    parser = DiscourseParser(input_dataset)
    parser.parse()

    output = open(sys.argv[6] + '/output.json', 'w')
    for relation in parser.relations:
        output.write('%s\n' % json.dumps(relation))
    output.close()

