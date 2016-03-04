"""
Main starting point.
Better docs coming.
"""
from utils.misc_utils import get_config, get_logger
from utils.resources import PDTBRelations
from utils.extractors import Word2Vec, OneHot, RandomVectors
from utils.models import CNN, RNN, LogisticRegression
from sys import argv

import numpy as np
np.random.seed(0)  # pylint: disable=E1101

# from joblib.memory import Memory Fix caches later
# MEMORY = Memory(cachedir='/tmp', verbose=logging.DEBUG)

RESOURCE_HANDLERS = {
    'conll16st-en-01-12-16-train': PDTBRelations,
    'conll16st-en-01-12-16-dev': PDTBRelations,
}

EXTRACTOR_HANDLERS = {
    'word2vec': Word2Vec,
    'onehot': OneHot,
    'random_vectors': RandomVectors
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
    resource_config = {x:y for x,y in resource_config.items() if x != 'name'}
    resource = resource_handler(**resource_config)
    return resource


def load_stored_model(model_path):
    logger.info("Loading stored model from {}".format(model_path))
    raise NotImplementedError()


def get_answers(instances):
    return list(instances.get_correct())

# Turn this on when you don't want to recompute features all the time
# @MEMORY.cache
def extract_features(feat_config, instances):
    """
    Data should be of type PDTBRelations for now. I should generalize this.
    Returns with dimensionality:
    sentences x words x n_features
    """
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

    return instances.get_feature_tensor(extractors)


def get_model(model_name):
    model = MODEL_HANDLERS[model_name]
    return model


def run(config):
    logger.info("Setting up...")
    # Load resources

    if config['train']:
        training_data = load_resource(config['resources'][config['train']])
        logger.debug("Training data classes: {}".format(training_data.y_indices))
        correct = get_answers(training_data)
        extracted_features = extract_features(config['feature_extraction'], training_data)
        model_class = get_model(config['model'])
        model = model_class(n_words=extracted_features.shape[1],
                            n_features=extracted_features.shape[2],
                            n_classes=len(training_data.y_indices),
                            **config['models'][config['model']])

        model.train(extracted_features, correct)
        logger.info("Finished training!")

    if config['test']:
        test_data = load_resource(config['resources'][config['test']])
        extracted_features = extract_features(config['feature_extraction'], test_data)
        model_class = get_model(config['model'])
        model = model_class(n_words=extracted_features.shape[1],
                            n_features=extracted_features.shape[2],
                            n_classes=len(test_data.y_indices),
                            **config['models'][config['model']])
        results = model.test(extracted_features)
        test_data.store_results(results, 'resources/conll16st-en-zh-dev-train_LDC2016E50/conll16st-en-01-12-16-dev/relations.json')
        logger.info("Finished testing!")


if __name__ == '__main__':
    if len(argv) == 1:
        config_ = get_config('config.yaml')
    else:
        config_ = get_config(argv[1])
    logger = get_logger(__name__, config=config_['logging'])
    run(config_)
