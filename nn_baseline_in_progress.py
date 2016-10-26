
from tensorflow.models.rnn import linear
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from .extractors import Word2Vec
class SNLIBaseline(Model):
    """
    Reimplementation of this thing:
    https://github.com/matpalm/snli_nn_tf/blob/master/nn_baseline.py
    """
    def __init__(self, n_features, n_classes, batch_size, epochs, word2vec_path, store_path=None, name=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.vocabulary_size = 3000
            self.sgd_learning_rate = 0.01
            self.sequence_length = 25
            self.hidden_dim = 100
            self.embedding_dim = 300
            self.batch_size = 10
            self.vocabulary_size = 300000
            self.optimizer = tf.train.GradientDescentOptimizer(self.sgd_learning_rate)
            self.num_epochs = 1
            self.momentum = 0
            self.train_x = tf.placeholder(tf.float32, [None, 1, n_features], name='x')
            self.train_y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
            self.train_embeddings = True
            self.word2vec_path = word2vec_path

        super().__init__()


    def train(self, feature_tensor, correct):
        s1_f = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])  # forward over s1
        s1_b = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])  # backwards over s1
        s2_f = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        s2_b = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length])
        inputs = [s1_f, s1_b, s2_f, s2_b]

        self.embeddings = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_dim]), name="embeddings")
        if self.word2vec_path:
            self.embeddings.assign(Word2Vec(word2vec_path, 'connective_token').data.syn0)

        embeddings *= np.asarray([0]*self.embedding_dim + [1]*((self.vocabulary_size-1)*self.embedding_dim)).reshape(self.vocabulary_size, self.embedding_dim)
        embedded_inputs = [self._embedded_sequence(s) for s in inputs]
        final_states = [self._final_state_of_rnn_over_embedded_sequence(idx, s) for idx, s in enumerate(embedded_inputs)]

        # concat these states into a single matrix
        # i.e. go from len=4 [(batch_size, hidden), ...] to single element
        concatted_final_states = tf.concat(1, final_states)
        

    def _embedded_sequence(self, sequence):
        # (batch_size, sequence_length, embedding_dim)
        embedded_inputs = tf.nn.embedding_lookup(self.embeddings, sequence)
        if not self.train_embeddings:
            embedded_inputs = tf.stop_gradient(embedded_inputs)
        # list of (batch_size, 1, embedding_dim)
        inputs = tf.split(1, self.sequence_length, embedded_inputs)
        # list of (batch_size, embedding_dim)
        inputs = [tf.squeeze(i) for i in inputs]
        return inputs

    def _final_state_of_rnn_over_embedded_sequence(self, idx, embedded_sequence):
        with tf.variable_scope("rnn_{}".format(idx)):
            gru = rnn_cell.GRUCell(hidden_dim)
            initial_state = gru.zero_state(self.batch_size, tf.float32)
            outputs, _states = rnn.rnn(gru, embedded_seq, initial_state=initial_state)
            return outputs[-1]

    def test(self, feature_tensor, correct):
        raise NotImplementedError()

    def store(self, store_path):
        raise NotImplementedError()
