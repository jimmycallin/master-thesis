from spacy import English
import yaml, logging, sys
import logging.config

def get_config(path):
    with open(path) as f:
        return yaml.load(f)

def get_logger(module_name, config=None):
    if config is not None:
        logging.config.dictConfig(config)
    return logging.getLogger(module_name)


en_model = None
def get_en_model():
    """
    This takes a while to load, so we make it a singleton.
    I know, singletons are bad, but the state never change for the object.
    """
    global en_model
    logger = get_logger(__name__)
    if en_model is None:
        logger.info("Loading spacy English model...")
        en_model = English(tagger=False,
                           parser=False,
                           entity=False,
                           matcher=False,
                           load_vectors=False)
    return en_model
