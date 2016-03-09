"""
TODO
"""
import abc, os
import tensorflow as tf
import numpy as np
from .misc_utils import get_logger

logger = get_logger(__name__)

class Model(metaclass=abc.ABCMeta):
    """
    TODO
    """
    def __init__(self):
        self.is_trained = False

    @abc.abstractmethod
    def train(self, feature_tensor, correct):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")

    @abc.abstractmethod
    def test(self, feature_tensor, correct):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")

    @abc.abstractmethod
    def store(self, store_path, session):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")


    @abc.abstractmethod
    def restore(self, store_path, session):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")


class CNN(Model):
    """
    TODO
    """
    def __init__(self, sgd_learning_rate):
        self.sgd_learning_rate = sgd_learning_rate

    def train(self, feature_tensor, correct):
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
        raise NotImplementedError()


class RNN(Model):
    """
    Reimplementation of this thing:
    https://github.com/matpalm/snli_nn_tf
    """
    def __init__(self, sgd_learning_rate):
        self.sgd_learning_rate = sgd_learning_rate

    def train(self, feature_tensor, correct):
        raise NotImplementedError()

    def test(self, feature_tensor, correct):
        raise NotImplementedError()

    def store(self, store_path):
        raise NotImplementedError()

from sklearn import svm
from sklearn.externals import joblib
class SVM(Model):
    def __init__(self, n_features, n_classes, kernel, store_path=None, name=None):
        self.kernel = kernel
        self.n_features = n_features
        self.n_classes = n_classes
        self.store_path = store_path
        self.model = None
        self.name = name

    def restore(self, store_path):
        logger.info("Restoring model from {}".format(store_path))
        return joblib.load(store_path)

    def store(self, model, store_path):
        logger.info("Storing model at {}".format(store_path))
        return joblib.dump(model, store_path)

    def train(self, feature_tensor, correct):
        logger.info("Training model...")
        squeezed = feature_tensor.squeeze(axis=1)
        clf = svm.SVC(kernel=self.kernel)
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

class LogisticRegression(Model):
    """
    Simple logreg model just to have some sort of baseline.
    """
    def __init__(self, n_features, n_classes, batch_size, epochs, store_path=None, name=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.n_features = n_features
            self.n_classes = n_classes
            self.batch_size = batch_size
            self.train_x = tf.placeholder(tf.float32, shape=(None, 1, n_features), name='x')
            self.train_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
            self.weights = tf.Variable(tf.random_normal((n_features, n_classes), stddev=0.01),
                                                        name='weights')
            self.epochs = epochs
            self.store_path = store_path
            self.name = name

        super(LogisticRegression, self).__init__()

    def restore(self, store_path, session):
        saver = tf.train.Saver()
        saver.restore(session, store_path)
        logger.info("Restored model from {}".format(store_path))

    def store(self, store_path, session):
        os.makedirs(os.path.join(*store_path.split("/")[:-1]), exist_ok=True)
        saver = tf.train.Saver([self.weights])
        saver.save(session, store_path)
        logger.debug("Stored model at {}".format(store_path))

    def massage_answers(self, correct):
        labels_dense = np.array(correct)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * self.n_classes
        labels_one_hot = np.zeros((num_labels, self.n_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        logger.debug("Converted dense vector to onehot with shape {}".format(labels_one_hot.shape))
        return labels_one_hot

    def train(self, feature_tensor, correct):
        """
        TODO
        """
        with self.graph.as_default():
            n_instances, _, _ = feature_tensor.shape
            assert n_instances == len(correct), \
                "Shape mismatch: feature tensor: {}, correct: {}".format(feature_tensor.shape,
                                                                         len(correct))

            correct_onehot = self.massage_answers(correct)
            logger.debug("Setting up training. feature_tensor has shape {},\
                          correct_onehot has shape {}".format(feature_tensor.shape,
                                                              correct_onehot.shape))

            squeezed = tf.squeeze(self.train_x, squeeze_dims=[1]) # Assume 3-dim tensor, need to be 2-dim
            p_y_given_x = tf.matmul(squeezed, self.weights, name='p_y_given_x')
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(p_y_given_x, self.train_y))
            train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


            start_i = range(0, n_instances, self.batch_size)
            end_i = range(self.batch_size, n_instances, self.batch_size)

            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
                logger.info("Starting training session...")
                for epoch_i in range(self.epochs):
                    logger.debug("Epoch: {}".format(epoch_i))
                    for start, end in zip(start_i, end_i):
                        logger.debug("Batch {}-{}".format(start, end))
                        sess.run(train_op, feed_dict={self.train_x: feature_tensor[start:end],
                                                      self.train_y: correct_onehot[start:end]})
                if self.store_path:
                    self.store(self.store_path, sess)
        logger.info("Training session done.")


    def test(self, feature_tensor):
        with self.graph.as_default():
            squeezed = tf.squeeze(self.train_x, squeeze_dims=[1]) # Assume 3-dim tensor, need to be 2-dim
            p_y_given_x = tf.matmul(squeezed, self.weights)
            predicted_classes = tf.argmax(p_y_given_x, 1)
            with tf.Session() as sess:
                self.restore(self.store_path, sess)

                init = tf.initialize_all_variables()
                sess.run(init)
                predicted_classes = sess.run(predicted_classes, feed_dict={self.train_x: feature_tensor})

        return predicted_classes
