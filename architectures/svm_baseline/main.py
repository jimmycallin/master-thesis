"""
Main starting point.
Better docs coming.
"""
from misc_utils import get_config, get_logger, timer
from resources import PDTBRelations
from extractors import CBOW
from model import SVM

import argparse

import numpy as np
np.random.seed(0)  # pylint: disable=E1101

EXTRACTOR_HANDLERS = {
    'cbow': CBOW,
}

MODEL_HANDLERS = {
    'svm': SVM
}

def load_resource(resource_config):
    logger.debug("Loading data from %s", resource_config['path'])
    resource_config = {x:y for x,y in resource_config.items() if x != 'name'}
    resource = PDTBRelations(**resource_config)
    return resource

def get_answers(instances):
    return list(instances.get_correct())

def extract_features(extract_config, instances):
    """
    Data should be of type PDTBRelations for now. I should generalize this.
    Returns with dimensionality:
    sentences x words x n_features
    """
    extractors = []
    # Sorting just makes sure they always end up in the same order,
    # Python's random hashing could mess this up
    for params in sorted(extract_config, key=lambda v: v['name']):
        extractor_params = {x: y for x,y in params.items() if x != 'name'}
        extractor = EXTRACTOR_HANDLERS[params['name']](**extractor_params)
        extractors.append(extractor)

    return instances.get_feature_tensor(extractors)


def get_model(model_name):
    model = MODEL_HANDLERS[model_name]
    return model


def run_experiment(config):
    logger.info("Setting up...")
    # Load resources

    train_time = timer()
    if config['train']:
        training_data = load_resource(config['resources']['training_data'])
        logger.debug("Training data classes: {}".format(training_data.y_indices))
        correct = get_answers(training_data)
        extracted_features = extract_features(config['extractors'], training_data)
        model_class = get_model(config['model']['name'])
        with train_time:
            model_config = {x:y for x,y in config['model'].items() if x != 'name'}
            model = model_class(n_features=extracted_features.shape[2],
                                n_classes=len(training_data.y_indices),
                                **model_config)

            model.train(extracted_features, correct)

        logger.info("Finished training!")

    test_time = timer()
    if config['test']:
        test_data = load_resource(config['resources']['test_data'])
        extracted_features = extract_features(config['extractors'], test_data)
        model_class = get_model(config['model']['name'])
        with test_time:
            model_config = {x:y for x,y in config['model'].items() if x != 'name'}
            model = model_class(n_features=extracted_features.shape[2],
                                n_classes=len(test_data.y_indices),
                                **model_config)
            predicted = model.test(extracted_features)

        test_data.store_results(predicted, config['test_output_path'])
        logger.info("Finished testing!")


if __name__ == '__main__':

    config = get_config('config.yaml')

    parser = argparse.ArgumentParser(description='Run experiment.')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--train-path', type=str, required=False)
    parser.add_argument('--dev-path', type=str, required=False)
    parser.add_argument('--test-path', type=str, required=False)
    parser.add_argument('--embedding-path', type=str, required=False)
    parser.add_argument('--model-store-path', type=str, required=False)
    parser.add_argument('--test-output-path', type=str, required=False)
    parser.add_argument('--svm-kernel', type=str, required=False)
    parser.add_argument('-c', type=float, required=False)

    args = parser.parse_args()

    config['train'] = args.train
    config['test'] = args.test
    if args.train_path:
        config['resources']['training_data']['path'] = args.train_path
    if args.dev_path:
        config['resources']['dev_data']['path'] = args.dev_path
    if args.test_path:
        config['resources']['test_data']['path'] = args.test_path
    if args.model_store_path:
        config['model']['store_path'] = args.model_store_path
    if args.test_output_path:
        config['test_output_path'] = args.test_output_path
    if args.svm_kernel:
        config['model']['kernel'] = args.svm_kernel
    if args.c:
        config['model']['c'] = args.c

    if args.embedding_path:
        for extractor in config['extractors']:
            extractor['path'] = args.embedding_path

    logger = get_logger(__name__, config=config['logging'])
    logger.info("Config: {}".format(config))
    run_experiment(config)
