from .misc_utils import get_config, get_logger, tokenize
from .pdtb_utils import DiscourseRelation
from collections import Counter, defaultdict
import json
import abc
import numpy as np

logger = get_logger(__name__)

class Resource(metaclass=abc.ABCMeta):
    def __init__(self, path, max_words_in_sentence, classes, padding):
        self.path = path
        self.max_words_in_sentence = max_words_in_sentence
        self.classes = sorted(classes)
        self.y_indices = {x: y for y, x in enumerate(self.classes)}
        self.instances = list(self._load_instances(path))
        self.padding = padding

    @abc.abstractmethod
    def _load_instances(self, path):
        raise NotImplementedError("This class must be subclassed.")


class PDTBRelations(Resource):
    def __init__(self, path, max_words_in_sentence, max_hierarchical_level, classes, separate_dual_classes, padding, filter_type=[]):
        self.max_hierarchical_level = max_hierarchical_level
        self.separate_dual_classes = separate_dual_classes
        self.filter_type = filter_type
        super(PDTBRelations, self).__init__(path, max_words_in_sentence, classes, padding)

    def _load_instances(self, path):
        with open(path) as file_:
            for line in file_:
                rel = DiscourseRelation(json.loads(line.strip()))
                if rel.relation_type() in self.filter_type:
                    continue
                if self.separate_dual_classes:
                    for splitted in rel.split_up_senses(max_level=self.max_hierarchical_level):
                        if len(splitted.senses()) > 1:
                            raise ValueError("n_senses > 1")
                        if len(splitted.senses()) == 1 and splitted.senses(max_level=self.max_hierarchical_level)[0] not in self.y_indices:
                            logger.debug("Sense class {} not in class list, skipping {}".format(splitted.senses(max_level=self.max_hierarchical_level)[0], splitted.relation_id()))
                            continue
                        yield splitted
                else:
                    all_classes_exist = all(r in self.y_indices for r in rel.senses(max_level=self.max_hierarchical_level))
                    if not all_classes_exist:
                        logger.debug("Sense {} classes not in class list, skipping {}".format(rel.senses(max_level=self.max_hierarchical_level), rel.relation_id()))
                        continue
                    yield rel

    def massage_sentence(self, sentence):
        if sentence is None:
            tokenized = ["NONE"]
        else:
            tokenized = tokenize(sentence)

        if self.max_words_in_sentence:
            tokenized = tokenized[:self.max_words_in_sentence]
        if self.padding:
            tokenized = tokenized + ['PADDING'] * (self.max_words_in_sentence - len(tokenized))

        return tokenized

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
                arg_tokenized = self.massage_sentence(arg_rawtext)
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
            senses = rel.senses(max_level=self.max_hierarchical_level)
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


    def calculate_accuracy(self, predicted):
        equal = 0
        gold = list(self.get_correct(indices=True))
        assert len(predicted) == len(gold)
        for p, g in zip(predicted, gold):
            assert isinstance(g, list)
            if p in g:
                equal += 1
        return equal / len(predicted)

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
        import json
        with open(store_path, 'w') as w:
            for rel in predicted_rels:
                w.write(json.dumps(rel) + '\n')
        logger.info("Stored predicted output at {}".format(store_path))


def official_scorer(prediction_file_path, gold_file_path):
    from .conll16st.scorer import evaluate_sense

    with open(gold_file_path) as file_:
        gold_rels = [json.loads(line) for line in file_]
    with open(prediction_file_path) as file_:
        predicted_rels = [json.loads(line) for line in file_]
    sense_cm = evaluate_sense(gold_rels, predicted_rels)
    print('Sense classification--------------')
    sense_cm.print_summary()


def evaluate_results(prediction_file_path, gold_file_path, print_report=True):
    official_scorer(prediction_file_path, gold_file_path)

    gold_rels = {}
    pred_rels = {}
    with open(gold_file_path) as gold_file, open(prediction_file_path) as pred_file:
        for line in gold_file:
            rel = DiscourseRelation(json.loads(line))
            gold_rels[rel.relation_id()] = rel

        for line in pred_file:
            rel = DiscourseRelation(json.loads(line))
            pred_rels[rel.relation_id()] = rel

    results = {}
    correct, incorrect, total = defaultdict(int), defaultdict(int), 0
    classes = set()

    type_correct = {'Explicit': 0, 'Implicit': 0, 'EntRel': 0, 'AltLex': 0}
    type_incorrect = type_correct.copy()

    for relation_id, pred_rel in pred_rels.items():
        gold_rel = gold_rels[relation_id]
        _ = [classes.add(s) for s in gold_rel.senses(max_level=3) + pred_rel.senses(max_level=3)]
        assert len(pred_rel.senses(max_level=3)) == 1
        total += 1
        pred_sense = pred_rel.senses(max_level=3)[0]
        if pred_sense in gold_rel.senses(max_level=3):
            correct[pred_sense] += 1
            type_correct[gold_rel.relation_type()] += 1
        else:
            # This only looks at the first gold sense, while there could be several.
            # This is how the official scorer does it, so let's just go with it.
            incorrect[gold_rel.senses(max_level=3)[0]] += 1
            type_incorrect[gold_rel.relation_type()] += 1

    # Fill in missing keys
    for cl in classes - set(incorrect.keys()):
        incorrect[cl] = 0
    for cl in classes - set(correct.keys()):
        correct[cl] = 0

    total_incorrect = sum(incorrect.values())
    total_correct = sum(correct.values())
    results['total_correct'] = total_correct
    results['total_incorrect'] = total_incorrect
    results['total_instances'] = total
    total_accuracy = total_correct / total if total != 0 else 0
    results['total_accuracy'] = total_accuracy

    report = ""
    report += "====== RESULTS =======\n"
    report += "Total: {} correct / {} ({})\n".format(results['total_correct'],
                                                     results['total_instances'],
                                                     results['total_accuracy'])
    report += "Specific classes:\n"
    report += "- - - - - - - - - - \n"
    results['classes'] = {}

    for class_ in classes:
        total_class = correct[class_] + incorrect[class_]
        class_accuracy = correct[class_] / total_class if total_class != 0 else 0
        results['classes'][class_] = {'correct': correct[class_],
                                      'total_class': total_class,
                                      'accuracy': class_accuracy}
        report += "{}: {} correct / {} ({})\n".format(class_,
                                                      results['classes'][class_]['correct'],
                                                      results['classes'][class_]['total_class'],
                                                      results['classes'][class_]['accuracy'])
    results['type'] = {}
    for rel_type in type_correct.keys():
        corr = type_correct[rel_type]
        incorr = type_incorrect[rel_type]
        acc = corr / (corr + incorr) if (corr + incorr) > 0 else 0
        results['type'][rel_type] = {'correct': corr,
                                     'total_class': corr + incorr,
                                     'accuracy': acc}

    report += "- - - - - - - - - - \n"
    if print_report:
        print(report)

    return results
