from os.path import join
from sys import path
import json
path.append("..")
from config import config

class DiscourseRelation():
    def __init__(self, dict_relation):
        self.raw = dict_relation

    def relation_id(self):
        return self.raw['ID']

    def doc_id(self):
        return self.raw['DocID']

    def senses(self, max_level=2):
        return [".".join(s.split(".")[:max_level]) for s in self.raw['Sense']]


    def relation_type(self):
        return self.raw['Type']

    def is_explicit(self):
        return self.relation_type == 'Explicit'

    #### ARG1 ####

    def arg1_text(self):
        return self.raw['Arg1']['RawText']

    def arg1_character_offsets(self):
        return self.raw['Arg1']['CharacterSpanList']

    #### ARG2 ####

    def arg2_text(self):
        return self.raw['Arg2']['RawText']

    def arg2_character_offsets(self):
        return self.raw['Arg2']['CharacterSpanList']

    #### CONNECTIVES ####

    def connective_token(self):
        return self.raw['Connective']['RawText']

    def connective_character_offsets(self):
        if self.is_explicit():
            return self.raw['Connective']['CharacterSpanList']['TokenList'][:2]
        else:
            return []

    def connective_token_document_offset(self):
        if self.is_explicit():
            return self.raw['Connective']['CharacterSpanList']['TokenList'][2]
        else:
            return None

    def connective_sentence_offset(self):
        if self.is_explicit():
            return self.raw['Connective']['CharacterSpanList']['TokenList'][3]
        else:
            return None

    def connective_token_sentence_offset(self):
        if self.is_explicit():
            return self.raw['Connective']['CharacterSpanList']['TokenList'][4]
        else:
            return None

    def __repr__(self):
        return "{}, {}: {}/{}".format(self.relation_id(), self.relation_type(),
                                  self.connective_token(), self.senses())


    def __str__(self):
        return "{} <-ARG1- {}: {}/{} -ARG2-> {}".format(self.arg1_text(),
                                                    self.relation_type(),
                                                    self.connective_token(),
                                                    self.senses(),
                                                    self.arg2_text())



def get_relations(relations_path):
    with open(relations_path) as f:
        for line in f:
            yield DiscourseRelation(json.loads(line.strip()))

def get_train_relations():
    relations_path = join(config['base_dir'],
                          config['train_dir'], 'relations.json')
    yield from get_relations(relations_path)


def get_test_relations():
    relations_path = join(config['base_dir'],
                          config['test_dir'], 'relations.json')
    yield from get_relations(relations_path)

if __name__ == '__main__':
    for disc_relation in get_train_relations():
        print(disc_relation)
