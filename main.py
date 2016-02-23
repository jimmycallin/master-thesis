from tools.misc_utils import get_config, get_logger
from tools.resources import ConllRelations
from tools.extractors import Word2Vec
from tools.models import CNN
import numpy as np
from sys import argv

resource_handlers = {
    'conll16st-en-01-12-16-train': ConllRelations,
}

feature_handlers = {
    'word2vec': Word2Vec
}

model_handlers = {
    'CNN': CNN
}

def load_resource(resource_dict):
    logger.debug("Loading {} from {}".format(resource_dict['name'],
                                             resource_dict['path']))
    resource_handler = resource_handlers[resource_dict['name']]
    resource = resource_handler(resource_dict['path'])
    return resource


def load_stored_model(model_path):
    logger.info("Loading stored model from {}".format(model_path))
    raise NotImplementedError()


def extract_features(feat_config, data):
    """
    Data should be of type ConllRelations for now. I should generalize this.
    Returns with dimensionality:
    sentences x words x embedding_size
    """
    feats = []
    config = {x:y for x,y in feat_config.items() if x != 'extractors'}
    # Sorting just makes sure they always end up in the same order,
    # Python's random hashing could mess this up
    for params in sorted(feat_config['extractors'], key=lambda v: v['name']):
        params = dict(params, **config)  # combine extractor specific feats with globals
        extractor = feature_handlers[params['name']](**params)
        feats.append(extractor.extract_features(data.get_sentences()))
    return np.concatenate(feats, axis=1)


def initiate_model(model_config):
    model = model_handlers[model_config['name']](**model_config)
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
        extracted_features = extract_features(config['feature_extraction'],
                                              training_data)
        model = initiate_model(config['model'])
        model.train(extracted_features)
        if 'store_path' in config['model']:
            model.store(config['model']['store_path'])
    else:
        model = load_stored_model(config['stored_model_path'])

    if config['test']:
        test_data = load_resource(config['test'])
        results = model.test(test_data)
        store_results(results, config)


    logger.info("Finished!")


if __name__ == '__main__':
    if len(argv) == 1:
        config = get_config('config.yaml')
    else:
        config = get_config(argv[1])
    logger = get_logger(__name__, config=config['logging'])
    run(config)
