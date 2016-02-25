from .misc_utils import get_config, get_logger, tokenize
from .pdtb_utils import get_relations
from collections import Counter
import abc
import numpy as np

logger = get_logger(__name__)

class Resource(metaclass=abc.ABCMeta):
    def __init__(self, path, max_words_in_sentence):
        self.path = path
        self.max_words_in_sentence = max_words_in_sentence
        self.instances = list(self._load_instances(path))
        self.n_instances = len(self.instances)
        self.y_indices = self._extract_y_indices(self.instances)

    @abc.abstractmethod
    def _extract_y_indices(self, instances):
        raise NotImplementedError("This class must be subclassed.")

    @abc.abstractmethod
    def _load_instances(self, path):
        raise NotImplementedError("This class must be subclassed.")


class PDTBRelations(Resource):
    def __init__(self, path, max_words_in_sentence):
        super(PDTBRelations, self).__init__(path, max_words_in_sentence)


    def _load_instances(self, path):
        return get_relations(self.path)

    def _extract_y_indices(self, instances):
        indices = set()
        for rel in instances:
            indices.add(str(rel.senses()))
        return {answer: index for index, answer in enumerate(sorted(indices))}

    def massage_sentence(self, sentence):
        tokenized = tokenize(sentence)[:self.max_words_in_sentence]
        padded = tokenized + ['PADDING'] * (self.max_words_in_sentence - len(tokenized))
        return padded

    def get_feature_tensor(self, extractors):
        rels_feats = []
        for rel in self.instances:
            arg1, arg2 = [self.massage_sentence(s) for s in [rel.arg1_text(),
                                                             rel.arg2_text()]]
            connective = [str(rel.connective_head()).strip().lower()]
            feats = []
            total_features_per_instance = 0
            for extractor in extractors:
                # These return matrices of shape (max_words, n_features)
                # We concatenate them on axis 1
                arg1_feats = extractor.extract_features(arg1)
                arg2_feats = extractor.extract_features(arg2)
                connective_feats = extractor.extract_features(connective)
                feats.append(np.concatenate([arg1_feats, arg2_feats, connective_feats], axis=0))
                total_features_per_instance += extractor.n_features

            rels_feats.append(np.concatenate(feats, axis=1))

        feature_tensor = np.array(rels_feats)
        assert_shape = (self.n_instances,
                        self.max_words_in_sentence * 2 + 1,
                        total_features_per_instance)
        assert feature_tensor.shape == assert_shape, \
                "Tensor shape mismatch. Is {}, should be {}".format(feature_tensor.shape, assert_shape)
        return feature_tensor

    def get_correct(self):
        """
        Returns answer indices.
        """

        for rel in self.instances:
            yield self.y_indices[str(rel.senses())]
