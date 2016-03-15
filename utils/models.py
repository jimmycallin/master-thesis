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


def test_cnn():
    return CNN()

import numpy as np

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))  # pylint:disable=E1101
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

class CNN():
    def __init__(self,
                 n_features=20,
                 n_classes=3,
                 embedding_dim=128,
                 filter_sizes=[3,4,5],
                 num_filters=128,
                 dropout_keep_prob=0,
                 l2_reg_lambda=0,
                 batch_size=64,
                 num_epochs=10,
                 evaluate_every=100,
                 checkpoint_every=100,
                 allow_soft_placement=True,
                 log_device_placement=False,
                 max_words_in_sentence=20,
                 vocab_size=300000,
                 store_path=None,
                 name=None):

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.max_words_in_sentence = max_words_in_sentence
        self.n_features = n_features
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.graph = tf.Graph()
        self.store_path = store_path

        with self.graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=self.allow_soft_placement,
                                          log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            self.sess = sess
            with self.sess.as_default():
                self.cnn = TextCNN(
                            sequence_length=self.max_words_in_sentence,
                            num_classes=self.n_classes,
                            vocab_size=self.vocab_size,
                            embedding_size=self.embedding_dim,
                            filter_sizes=self.filter_sizes,
                            num_filters=self.num_filters,
                            l2_reg_lambda=self.l2_reg_lambda)

                # Define Training procedure
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-4)
                grads_and_vars = optimizer.compute_gradients(self.cnn.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.merge_summary(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", self.cnn.loss)
                acc_summary = tf.scalar_summary("accuracy", self.cnn.accuracy)

                # Train Summaries
                self.train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                self.train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

                # Dev summaries
                self.dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                self.dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.saver = tf.train.Saver(tf.all_variables())

                # Initialize all variables
                sess.run(tf.initialize_all_variables())

    def train_step(self, x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          self.cnn.input_x: x_batch,
          self.cnn.input_y: y_batch,
          self.cnn.dropout_keep_prob: self.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = self.sess.run([self.train_op, self.global_step, self.train_summary_op, self.cnn.loss, self.cnn.accuracy],
                                                           feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def dev_step(self, x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        with self.graph.as_default(), self.sess.as_default():
            feed_dict = {
              self.cnn.input_x: x_batch,
              self.cnn.input_y: y_batch,
              self.cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = self.sess.run([self.global_step, self.dev_summary_op, self.cnn.loss, self.cnn.accuracy],
                                                            feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

    def train(self, feature_tensor, correct):
        feature_tensor = np.squeeze(feature_tensor, axis=1)
        correct = self.massage_answers(correct)
        with self.graph.as_default(), self.sess.as_default():
            x_train, x_dev = feature_tensor[:-1000], feature_tensor[-1000:]
            y_train, y_dev = correct[:-1000], correct[-1000:]

            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), self.batch_size, self.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                self.train_step(x_batch, y_batch)

                current_step = tf.train.global_step(self.sess, self.global_step)
                if current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    self.dev_step(x_dev, y_dev, writer=self.dev_summary_writer)
                    print("")
                if current_step % self.checkpoint_every == 0:
                    path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            if self.store_path:
                self.store(self.store_path, self.sess)

    def test(self, feature_tensor):
        feature_tensor = np.squeeze(feature_tensor, axis=1) # Assume 3-dim tensor, need to be 2-dim
        with self.graph.as_default(), self.sess.as_default():
            x_test = feature_tensor
            self.restore(self.store_path, self.sess)
            # Training loop. For each batch...

            feed_dict = {
              self.cnn.input_x: x_test,
              self.cnn.dropout_keep_prob: 1.0
            }
            step, predictions = self.sess.run([self.global_step, self.cnn.predictions], feed_dict)
        return predictions


    def massage_answers(self, correct):
        labels_dense = np.array(correct)
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * self.n_classes
        labels_one_hot = np.zeros((num_labels, self.n_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        logger.debug("Converted dense vector to onehot with shape {}".format(labels_one_hot.shape))
        return labels_one_hot

    def restore(self, store_path, session):
        saver = tf.train.Saver()
        saver.restore(session, store_path)
        logger.info("Restored model from {}".format(store_path))

    def store(self, store_path, session):
        os.makedirs(os.path.join(*store_path.split("/")[:-1]), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(session, store_path)
        logger.debug("Stored model at {}".format(store_path))


import time
import datetime
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


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
