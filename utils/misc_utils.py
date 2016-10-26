"""
Contains miscellaneous helper functions
"""

import logging
import logging.config
import yaml
import spacy
from time import time

def get_config(path):
    """
    Returns YAML configuration from path.
    """
    with open(path) as file_:
        return yaml.load(file_)

def get_logger(module_name, config=None):
    """
    Returns global logger.
    Params:
    - module_name: The name of the logger, preferably module name.
    - config: A dict of logger configuration.
    """
    if config is not None:
        logging.config.dictConfig(config)
    return logging.getLogger(module_name)


EN_MODEL = None
def get_en_model():
    """
    This takes a while to load, so we make it a singleton.
    I know, singletons are bad, but the state never change for the object.
    """
    global EN_MODEL  # pylint: disable=W0603
    logger = get_logger(__name__)
    if EN_MODEL is None:
        logger.debug("Loading spacy English model...")
        EN_MODEL = spacy.load(tagger=False,
                              parser=False,
                              entity=False,
                              matcher=False,
                              load_vectors=False)
    return EN_MODEL


def tokenize(sentence):
    """
    Returns tokenized string.
    """
    en_model = get_en_model()
    return [w.lower_ for w in en_model(sentence)]

class timer():
    def __init__(self):
        self.elapsed_time = 0

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(self, type_, value, traceback):
        self.elapsed_time = time() - self.start_time
        return
