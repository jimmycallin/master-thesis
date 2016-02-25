"""
Main starting point.
Better docs coming.
"""

from utils.misc_utils import get_config, get_logger
from utils.resources import PDTBRelations
from utils.extractors import Word2Vec
from utils.models import CNN, RNN, LogisticRegression
import numpy as np
from sys import argv
from joblib.memory import Memory
import logging

MEMORY = Memory(cachedir='/tmp', verbose=logging.DEBUG)

RESOURCE_HANDLERS = {
    'conll16st-en-01-12-16-train': PDTBRelations,
}

EXTRACTOR_HANDLERS = {
    'word2vec': Word2Vec
}

MODEL_HANDLERS = {
    'cnn': CNN,
    'nn_baseline': RNN,
    'logistic_regression': LogisticRegression
}

def load_resource(resource_config):
    """
    TODO
    """
    logger.debug("Loading {} from {}".format(resource_config['name'],
                                             resource_config['path']))
    resource_handler = RESOURCE_HANDLERS[resource_config['name']]
    resource = resource_handler(resource_config['path'], resource_config['max_words_in_sentence'])
    return resource


def load_stored_model(model_path):
    logger.info("Loading stored model from {}".format(model_path))
    raise NotImplementedError()

# Turn this on when you don't want to recompute features all the time
# @MEMORY.cache
def extract_features(feat_config, instances):
    """
    Data should be of type PDTBRelations for now. I should generalize this.
    Returns with dimensionality:
    sentences x words x n_features
    """
    y = list(instances.get_correct())
    extractors = []
    extract_config = {x:y for x, y in feat_config.items() if x != 'extractors'}
    # Sorting just makes sure they always end up in the same order,
    # Python's random hashing could mess this up
    for params in sorted(feat_config['extractors'], key=lambda v: v['name']):
        if params['name'] == 'word2vec' and config_['development_mode']:
            params['random_vectors_only'] = True

        params = {**params, **extract_config}  # combine extractor specific feats with globals
        extractor = EXTRACTOR_HANDLERS[params['name']](**params)
        extractors.append(extractor)

    return instances.get_feature_tensor(extractors), y


def get_model(model_name):
    model = MODEL_HANDLERS[model_name]
    return model


def store_results(results, config):
    """
    Don't forget to use the official scoring script here.
    """
    report = """TEST RESULTS
    ...
    """
    logger.info(report)
    with open(config['store_results'], 'a') as w:
        for line in report.split("\n"):
            w.write(line + "\n")
    logger.info("Stored test results in {}".format(config['store_test_results']))


def run(config):
    logger.info("Setting up...")
    # Load resources
    if config['train']:
        training_data = load_resource(config['resources']['training_data'])
        logger.debug("Training data classes: {}".format(training_data.y_indices))
        extracted_features, correct = extract_features(config['feature_extraction'],
                                                       training_data)
        model_class = get_model(config['model'].pop('name'))
        model = model_class(n_words=extracted_features.shape[1],
                            n_features=extracted_features.shape[2],
                            n_classes=len(training_data.y_indices),
                            **config['model'])
        model.train(extracted_features, correct)
        if 'store_path' in config['model']:
            pass
            #model.store(config['model']['store_path'])
    else:
        model = load_stored_model(config['stored_model_path'])

    if config['test']:
        test_data = load_resource(config['resources']['test_data'])
        results = model.test(test_data)
        store_results(results, config)


    logger.info("Finished!")


if __name__ == '__main__':
    if len(argv) == 1:
        config_ = get_config('config.yaml')
    else:
        config_ = get_config(argv[1])
    logger = get_logger(__name__, config=config_['logging'])
    run(config_)
