from .misc_utils import get_config, get_logger, tokenize
from .pdtb_utils import DiscourseRelation
from collections import Counter
import json
import abc
import numpy as np

logger = get_logger(__name__)

class Resource(metaclass=abc.ABCMeta):
    def __init__(self, path, max_words_in_sentence, classes):
        self.path = path
        self.max_words_in_sentence = max_words_in_sentence
        self.instances = list(self._load_instances(path))
        self.classes = sorted(classes)
        self.y_indices = {x: y for y, x in enumerate(self.classes)}

    @abc.abstractmethod
    def _load_instances(self, path):
        raise NotImplementedError("This class must be subclassed.")


class PDTBRelations(Resource):
    def __init__(self, path, max_words_in_sentence, max_hierarchical_level, classes, separate_dual_classes):
        self.max_hierarchical_level = max_hierarchical_level
        self.separate_dual_classes = separate_dual_classes
        super(PDTBRelations, self).__init__(path, max_words_in_sentence, classes)

    def _load_instances(self, path):
        with open(path) as file_:
            for line in file_:
                rel = DiscourseRelation(json.loads(line.strip()))
                if self.separate_dual_classes:
                    for splitted in rel.split_up_senses():
                        yield splitted
                else:
                    yield rel

    def massage_sentence(self, sentence):
        tokenized = tokenize(sentence)[:self.max_words_in_sentence]
        padded = tokenized + ['PADDING'] * (self.max_words_in_sentence - len(tokenized))
        return padded

    def get_feature_tensor(self, extractors):
        rels_feats = []
        n_instances = 0
        for rel in self.instances:
            n_instances += 1
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
        assert_shape = (n_instances,
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
            yield self.y_indices[str(rel.senses(max_level=self.max_hierarchical_level))]


    def store_results(self, results):
        """
        Don't forget to use the official scoring script here.
        """
        text_results = [self.classes[res] for res in results]
        # Load test file
        # Insert results to test file
        # Deal with multiple instances somehow
        for text_result, rel in zip(text_results, self.instances):
            rel.set_senses(eval(text_result))  # turn string representation into list instance first
            if rel.is_explicit():
                rel.set_relation_type('Explicit')
            else:
                rel.set_relation_type('Implicit')

        # Merge instances
        from collections import defaultdict
        instances = defaultdict(list)
        for rel in self.instances:
            instances[rel.relation_id()].append(rel)

        merged = []
        for rel_id in sorted(instances.keys()):
            rels = instances[rel_id]
            senses = [sense for rel in rels for sense in rel.senses()]
            merged_rel = DiscourseRelation(rels[0].raw.copy())
            merged_rel.set_senses(senses)
            merged.append(merged_rel)

        # Store test file
        import json
        with open('test.json', 'w') as w:
            for rel in merged:
                w.write(json.dumps(rel.raw) + '\n')
        logger.info("Stored test file at test.json")
        # Compare with gold standard
