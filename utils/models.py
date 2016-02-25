"""
TODO
"""
import tensorflow as tf
import numpy as np
from .misc_utils import get_logger

logger = get_logger(__name__)

class Model:
    """
    TODO
    """
    def train(self, feature_tensor, correct):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")

    def test(self, feature_tensor, correct):
        """
        TODO
        """
        raise NotImplementedError("This class must be subclassed")

    def store(self, store_path):
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

class LogisticRegression(Model):
    """
    Simple logreg model just to have some sort of baseline.
    """
    def __init__(self, n_words, n_features, n_classes, batch_size, epochs):
        self.n_words = n_words
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.train_x = tf.placeholder(tf.float32, shape=(None, n_words, n_features), name='x')
        self.train_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
        self.weights = tf.Variable(tf.random_normal((n_words*n_features, n_classes), stddev=0.01))
        self.epochs = epochs

    def train(self, feature_tensor, correct):
        """
        TODO
        """
        correct_onehot = dense_to_one_hot(correct, self.n_classes)
        logger.debug("Starting training. feature_tensor has shape {}, correct_onehot has shape {}".format(feature_tensor.shape,
                                                                                                          correct_onehot.shape))
        n_instances, n_words, n_features = feature_tensor.shape
        assert n_instances == len(correct), \
            "Shape mismatch: feature tensor: {}, correct: {}".format(feature_tensor.shape,
                                                                     len(correct))
        reshaped = tf.squeeze(tf.reshape(self.train_x, [-1, 1, n_words * n_features]))
        p_y_given_x = tf.matmul(reshaped, self.weights)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(p_y_given_x, self.train_y))
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)


        start_i = range(0, n_instances, self.batch_size)
        end_i = range(self.batch_size, n_instances, self.batch_size)

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for _ in range(self.epochs):
                for start, end in zip(start_i, end_i):
                    logger.debug("Batch {}-{}".format(start, end))
                    sess.run(train_op, feed_dict={self.train_x: feature_tensor[start:end],
                                                  self.train_y: correct_onehot[start:end]})
        logger.info("Model trained.")

    def test(self, feature_tensor, correct):
        #predicted_classes = tf.argmax(p_y_given_x, 1)
        raise NotImplementedError()

    def store(self, store_path):
        raise NotImplementedError()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.array(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    logger.debug("Converted dense vector to onehot with shape {}".format(labels_one_hot.shape))
    return labels_one_hot
