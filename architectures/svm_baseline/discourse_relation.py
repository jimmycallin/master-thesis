"""
Class for dealing with the discourse relations.
"""

class DiscourseRelation():
    def __init__(self, dict_relation):
        self.raw = dict_relation

    def relation_id(self):
        return self.raw['ID']

    def doc_id(self):
        return self.raw['DocID']

    def senses(self, max_level=3):
        """
        Removing duplicate senses, if there are any
        """
        return sorted(set([".".join(s.split(".")[:max_level]) for s in self.raw['Sense']]))

    def set_senses(self, senses):
        assert isinstance(senses, list), "Senses is of instance {}".format(type(senses))
        self.raw['Sense'] = senses

    def relation_type(self):
        return self.raw['Type']

    def set_relation_type(self, rel_type):
        assert rel_type in {'Implicit', 'Explicit'}
        self.raw['Type'] = rel_type

    def is_explicit(self):
        return self.relation_type == 'Explicit' or self.relation_type != ''

    def to_output_format(self, sense, rel_type):
        output = {'Arg1': {'TokenList': self.arg1_token_document_offsets(),
                           'RawText': self.arg1_text()},
                  'Arg2': {'TokenList': self.arg2_token_document_offsets(),
                           'RawText': self.arg2_text()},
                  'Connective': {'TokenList': [],
                                 'RawText': self.connective_token()},
                  'DocID': self.doc_id(),
                  'Sense': [sense],
                  'Type': rel_type,
                  'ID': self.relation_id()}
        return output

    def split_up_senses(self):
        for sense in self.senses():
            new_rel = DiscourseRelation(self.raw.copy())
            new_rel.set_senses([sense])
            yield new_rel

    #### ARG1 ####

    def arg1_text(self):
        return self.raw['Arg1']['RawText']

    def arg1_character_offsets(self):
        return self.raw['Arg1']['CharacterSpanList']

    def arg1_token_document_offsets(self):
        return [token[2] for token in self.raw['Arg1']['TokenList']]

    #### ARG2 ####

    def arg2_text(self):
        return self.raw['Arg2']['RawText']

    def arg2_character_offsets(self):
        return self.raw['Arg2']['CharacterSpanList']

    def arg2_token_document_offsets(self):
        return [token[2] for token in self.raw['Arg2']['TokenList']]

    #### CONNECTIVES ####

    def connective_token(self):
        token = self.raw['Connective']['RawText']
        if token == "":
            return None
        elif self.raw['Type'] == 'Implicit':
            return None
        else:
            return token

    def connective_character_offsets(self):
        return [token[:2] for token in self.raw['Connective']['TokenList']]

    def connective_token_document_offset(self):
        return [token[2] for token in self.raw['Connective']['TokenList']]

    def connective_sentence_offset(self):
        return [token[3] for token in self.raw['Connective']['TokenList']]

    def connective_token_sentence_offset(self):
        if self.is_explicit():
            return self.raw['Connective']['TokenList'][4]
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
