import numpy as np
from misc_utils import get_logger  # pylint: disable=E0401
import hashlib

logger = get_logger(__name__)

class EmptyData():
    def __init__(self):
        self.vocab = {}
        self.vector_size = 300


class RandomVectors():
    """
    Keeps track of randomly created vectors.
    The vectors are deterministically created based on the word,
    but we still store them to make retrieval faster.
    This has a negative effect on memory usage.
    """
    def __init__(self, argument, dimensionality):
        self.argument = argument
        self.vocab = {}
        self.n_features = dimensionality

    def extract_features(self, sentence):
        feats = np.array([self[w] for w in sentence])
        assert feats.shape == (len(sentence), self.n_features)
        return feats

    def __getitem__(self, w):
        if w not in self.vocab:
            # Setting hashing state ensures we have the same random vector for each word between runs
            hsh = hashlib.md5()
            hsh.update(w.encode())
            seed = int(hsh.hexdigest(), 16) % 4294967295  # highest number allowed by seed
            state = np.random.RandomState(seed)  # pylint: disable=E1101
            self.vocab[w] = state.randn(self.n_features)  # pylint: disable=E1101
        return self.vocab[w]

_word_embedding_data = None

class WordEmbedding():
    """
    This loads WordEmbedding vectors from binary or text file format, as created
    by the Google News corpus.
    The text format looks like this:
    n_words, n_dimensions
    word1 1.0 1.0 1.0 ...
    word2 0.5 0.1 0.2 ...
    etc.
    """
    def __init__(self, path, argument, is_binary):
        """
        Loads word vectors for words in vocab
        """
        self.path = path
        self.argument = argument
        self.data = self._load_from_file(self.path, is_binary=is_binary)

        self.n_embeddings = len(self.data.wv.vocab)
        self.n_features = self.data.vector_size
        self.random_vectors = RandomVectors(argument, self.n_features)

    def extract_features(self, sentence):
        """
        It is constrained to a specific number of words per sentence.
        If the sentence is too long, we cut it off at the end.
        If it is too short, we add zero vectors to it.
        Returns features according to:
            n_words x n_features
        """
        feats = np.array([self[w] for w in sentence])
        assert feats.shape == (len(sentence), self.n_features)
        return feats


    def _load_from_file(self, path, is_binary=True):
        """
        Loads the WordEmbedding binaries.
        Loading the binaries takes forever, so we only load it into memory once as a singleton.
        """
        # We keep the
        global _word_embedding_data  # pylint: disable=W0603
        if _word_embedding_data is None:
            logger.info("Loading WordEmbedding matrix, this will take a while...")
            import gensim
            _word_embedding_data = gensim.models.Word2Vec.load_word2vec_format(path, binary=is_binary)

        logger.debug("WordEmbedding matrix loaded")
        return _word_embedding_data

    def __getitem__(self, w):
        if w in self.data.wv.vocab:
            return self.data[w]
        else:
            return self.random_vectors[w]

class CBOW(WordEmbedding):
    """
    Creates a centroid vector by averaging the word embeddings of the sentence.
    """
    def __init__(self, path, argument, is_binary):
        super().__init__(path, argument, is_binary)

    def extract_features(self, sentence):
        feats = np.mean(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats
