import tensorflow as tf
from .misc_utils import get_logger

logger = get_logger(__name__)

class Model:
    def train(self, feature_tensor, correct):
        raise NotImplementedError("This class must be subclassed")

    def test(self, feature_tensor, correct):
        raise NotImplementedError("This class must be subclassed")

    def store(self, store_path):
        raise NotImplementedError("This class must be subclassed")

class CNN(Model):
    def __init__(self, sgd_learning_rate, **kwargs):
        self.learning_rate = sgd_learning_rate

    def train(self, feature_tensor):
        """
        feature_tensor: 3D tensor of features, each axis corresponding to:
                            1. Word features
                            2. Words
                            3. Sentences
        """
        logger.info("Starting training...")
        raise NotImplementedError()

    def test(self, feature_tensor, correct):
        """
        feature_tensor: 3D tensor of features, each axis corresponding to:
                            1. Word features
                            2. Words
                            3. Training instance
        correct: 1D tensor of correct answer for each training instance.
        """
        logger.info("Starting testing of model...")
        raise NotImplementedError()

    def store(self, store_path):

        self.store_path = store_path
        raise NotImplementedError()
