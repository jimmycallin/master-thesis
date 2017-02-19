from misc_utils import get_config, get_logger, tokenize
from discourse_relation import DiscourseRelation
from collections import Counter, defaultdict
import json
import abc
import numpy as np
from os.path import join
import os

logger = get_logger(__name__)

class Resource(metaclass=abc.ABCMeta):
    def __init__(self, path, classes):
        self.path = path
        self.classes = sorted(classes)
        self.y_indices = {x: y for y, x in enumerate(self.classes)}
        self.instances = list(self._load_instances(path))

    @abc.abstractmethod
    def _load_instances(self, path):
        raise NotImplementedError("This class must be subclassed.")


class PDTBRelations(Resource):
    def __init__(self, path, classes, separate_dual_classes, filter_type=None, skip_missing_classes=True):
        self.skip_missing_classes = skip_missing_classes
        self.separate_dual_classes = separate_dual_classes
        self.filter_type = [] if filter_type is None else filter_type
        super().__init__(path, classes)

    def _load_instances(self, path):
        with open(join(path, 'relations.json')) as file_:
            for line in file_:
                rel = DiscourseRelation(json.loads(line.strip()))
                if (self.filter_type != [] or self.filter_type is not None) and rel.relation_type() not in self.filter_type:
                    continue
                if self.separate_dual_classes:
                    for splitted in rel.split_up_senses():
                        if len(splitted.senses()) > 1:
                            raise ValueError("n_senses > 1")
                        if len(splitted.senses()) == 1 and splitted.senses()[0] not in self.y_indices:
                            if self.skip_missing_classes:
                                logger.debug("Sense class {} not in class list, skipping {}".format(splitted.senses()[0], splitted.relation_id()))
                                continue
                        yield splitted
                else:
                    a_class_exist = any(r in self.y_indices for r in rel.senses())
                    if not a_class_exist:
                        if self.skip_missing_classes:
                            logger.debug("Sense {} classes not in class list, skipping {}".format(rel.senses(), rel.relation_id()))
                            continue
                    yield rel

    def get_feature_tensor(self, extractors):
        rels_feats = []
        n_instances = 0
        last_features_for_instance = None
        for rel in self.instances:
            n_instances += 1
            feats = []
            total_features_per_instance = 0
            for extractor in extractors:
                # These return matrices of shape (1, n_features)
                # We concatenate them on axis 1
                arg_rawtext = getattr(rel, extractor.argument)()
                arg_tokenized = tokenize(arg_rawtext)
                arg_feats = extractor.extract_features(arg_tokenized)
                feats.append(arg_feats)
                total_features_per_instance += extractor.n_features
            if last_features_for_instance is not None:
                # Making sure we have equal number of features per instance
                assert total_features_per_instance == last_features_for_instance
            rels_feats.append(np.concatenate(feats, axis=1))

        feature_tensor = np.array(rels_feats)
        assert_shape = (n_instances, 1, total_features_per_instance)
        assert feature_tensor.shape == assert_shape, \
                "Tensor shape mismatch. Is {}, should be {}".format(feature_tensor.shape, assert_shape)
        return feature_tensor

    def get_correct(self, indices=True):
        """
        Returns answer indices.
        """

        for rel in self.instances:
            senses = rel.senses()
            if self.separate_dual_classes:
                if indices:
                    yield self.y_indices[senses[0]]
                else:
                    yield senses[0]
            else:
                ys = [self.y_indices[sense] for sense in senses]
                if indices:
                    yield ys
                else:
                    yield senses

    def store_results(self, results, store_path):
        """
        Don't forget to use the official scoring script here.
        """
        text_results = [self.classes[res] for res in results]
        # Load test file
        # Output json object with results
        # Deal with multiple instances somehow
        predicted_rels = []
        for text_result, rel in zip(text_results, self.instances):
            if rel.is_explicit():
                rel_type = 'Explicit'
            else:
                rel_type = 'Implicit'
            predicted_rels.append(rel.to_output_format(text_result, rel_type))  # turn string representation into list instance first

        # Store test file
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        with open(join(store_path, 'output.json'), 'w') as w:
            for rel in predicted_rels:
                w.write(json.dumps(rel) + '\n')
        logger.info("Stored predicted output at {}".format(store_path))
