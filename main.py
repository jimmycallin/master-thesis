"""
Main starting point.
Better docs coming.
"""
from utils.misc_utils import get_config, get_logger, timer
from utils.resources import PDTBRelations, evaluate_results
from utils.extractors import Word2Vec, OneHot, RandomVectors, CBOW, BagOfWords, RandomCBOW, VocabIndices
from utils.models import CNN, SVM, LogisticRegression
from expy import Project
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
    'random_vectors': RandomVectors,
    'random_cbow': RandomCBOW,
    'cbow': CBOW,
    'bag_of_words': BagOfWords,
    'vocab_indices': VocabIndices
}

MODEL_HANDLERS = {
    'cnn': CNN,
    'svm': SVM,
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
        extractor = EXTRACTOR_HANDLERS[params['name']](**params)
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
        training_data = load_resource(config['resources'][config['train']])
        logger.debug("Training data classes: {}".format(training_data.y_indices))
        correct = get_answers(training_data)
        extracted_features = extract_features(config['extractors'], training_data)
        model_class = get_model(config['model']['name'])
        with train_time:
            model = model_class(n_features=extracted_features.shape[2],
                                n_classes=len(training_data.y_indices),
                                **config['model'])

            model.train(extracted_features, correct)

        logger.info("Finished training!")

    test_time = timer()
    if config['test']:
        test_data = load_resource(config['resources'][config['test']])
        extracted_features = extract_features(config['extractors'], test_data)
        model_class = get_model(config['model']['name'])
        with test_time:
            model = model_class(n_features=extracted_features.shape[2],
                                n_classes=len(test_data.y_indices),
                                **config['model'])
            predicted = model.test(extracted_features)

        gold = np.array(list(test_data.get_correct()))
        test_data.store_results(predicted, config['test_output_path'])

        results = evaluate_results(config['test_output_path'],
                                   config['resources'][config['test']]['path'],
                                   print_report=config['print_report'])
        logger.info("Finished testing!")
        return results, train_time.elapsed_time, test_time.elapsed_time


if __name__ == '__main__':
    base_config = get_config('config.yaml')
    model_config = get_config(argv[1])
    config_ = {**base_config, **model_config}
    logger = get_logger(__name__, config=config_['logging'])

    project_config = {x:y for x,y in base_config.items() if x not in {'project_name',
                                                                      'description',
                                                                      'results_db_uri',
                                                                      'logging',
                                                                      'deploy'}}
    project = Project(project_name=config_['project_name'],
                      description=config_['description'], # Project description
                      project_config=project_config,  # Base configuration for project
                      mongodb_uri=config_['results_db_uri'],
                      force_clean_repo=False)  # Crash program if git status doesn't return clean repo


    experiment_config = {x:y for x,y in config_.items() if x not in {'project_name',
                                                                     'description',
                                                                     'results_db_uri',
                                                                     'author',
                                                                     'experiment_name',
                                                                     'description',
                                                                     'logging',
                                                                     'deploy',
                                                                     'tags'}}
    with project.new_experiment(config=experiment_config,
                                author=config_.get('author', None),
                                experiment_name=config_.get('experiment_name', None),
                                description=config_.get('description', None),
                                tags=config_['tags']) as experiment:  # This starts execution timer

        results, train_time, test_time = run_experiment(config_)

        experiment.experiment_results = results
        experiment.train_time = train_time
        experiment.test_time = test_time

        logger.info("Stored model configuration, commit ID, execution time, and test results at {}".format(project.db_uri))
