"""
Model to use for text classification.
"""

from sklearn import svm
from sklearn.externals import joblib
from misc_utils import get_logger
import os
logger = get_logger(__name__)

class SVM():
    def __init__(self, n_features, n_classes, kernel, c, store_path):
        self.is_trained = False
        self.kernel = kernel
        self.c = c
        self.n_features = n_features
        self.n_classes = n_classes
        self.store_path = store_path
        self.model = None

    def restore(self, store_path):
        logger.info("Restoring model from %s", store_path)
        return joblib.load(os.path.join(store_path, 'model.pkg'))

    def store(self, model, store_path):
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        logger.info("Storing model at %s", store_path)
        return joblib.dump(model, os.path.join(store_path, 'model.pkg'))

    def train(self, feature_tensor, correct):
        logger.info("Training model...")
        squeezed = feature_tensor.squeeze(axis=1)
        clf = svm.SVC(kernel=self.kernel, C=self.c)
        model = clf.fit(squeezed, correct)
        if self.store_path:
            self.store(model, self.store_path)
        self.model = model
        logger.info("Training session done")

    def test(self, feature_tensor):
        logger.info("Loading model...")
        self.model = self.restore(self.store_path)
        logger.info("Testing model...")
        squeezed = feature_tensor.squeeze(axis=1)
        return self.model.predict(squeezed)
