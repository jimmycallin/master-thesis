import numpy as np
import abc
from .misc_utils import get_logger  # pylint: disable=E0401
import hashlib

logger = get_logger(__name__)

class Extractor(metaclass=abc.ABCMeta):
    """
    Base class with expected functions for any Extractor subclass.
    """
    def __init__(self, argument):
        self.argument = argument

    @abc.abstractmethod
    def extract_features(self, sentences):
        raise NotImplementedError("This must be subclassed.")

class EmptyData():
    def __init__(self):
        self.vocab = {}
        self.vector_size = 300


class RandomVectors(Extractor):
    def __init__(self, dimensionality, argument, **kwargs):
        super().__init__(argument)
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


_word2vec_data = None

class Word2Vec(Extractor):
    """
    This loads Word2Vec vectors from binary file format, as created
    by the Google News corpus. You should download that first.
    """
    def __init__(self, path, argument, **kwargs):
        """
        Loads word vectors for words in vocab
        """
        super().__init__(argument)
        self.path = path
        self.data = self._load_from_binary_format(self.path)

        self.n_embeddings = len(self.data.vocab)
        self.n_features = self.data.vector_size
        self.random_vectors = RandomVectors(self.n_features, argument)

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


    def _load_from_binary_format(self, path):
        """
        Loads the Word2Vec binaries.
        Loading the binaries takes forever, so we only load it into memory once as a singleton.
        """
        global _word2vec_data  # pylint: disable=W0603
        if _word2vec_data is None:
            logger.info("Loading Word2Vec matrix, this will take a while...")
            import gensim
            _word2vec_data = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)

        logger.debug("Word2Vec matrix loaded")
        return _word2vec_data

    def __getitem__(self, w):
        if w in self.data.vocab:
            return self.data[w]
        else:
            return self.random_vectors[w]


class CBOW(Word2Vec):
    def __init__(self, path, argument, **kwargs):
        super().__init__(path, argument, **kwargs)

    def extract_features(self, sentence):
        feats = np.mean(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats

class RandomCBOW(RandomVectors):
    def __init__(self, dimensionality, **kwargs):
        super().__init__(dimensionality, **kwargs)

    def extract_features(self, sentence):
        feats = np.mean(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats

class OneHot(Extractor):
    def __init__(self, vocab_indices_path, max_vocab_size, argument, **kwargs):
        super().__init__(argument)
        self.vocab = self._read_vocab_indices(vocab_indices_path, max_vocab_size)
        self.n_features = len(self.vocab) + 1 # last one for oov

    def _read_vocab_indices(self, vocab_indices_path, max_vocab_size):
        """
        Assumes vocab is ordered according to frequency
        """
        vocab = {}
        with open(vocab_indices_path) as f:
            for i, line in enumerate(f):
                if i > max_vocab_size:
                    break
                if line.strip() in vocab:
                    raise ValueError("{} appears twice in vocab".format(line))
                vocab[line.strip()] = i
        logger.debug("Loaded vocabulary index. Size of vocab: {}".format(len(vocab)))
        return vocab

    def get(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return len(self.vocab) # oov

    def extract_features(self, sentence):
        feat_matrix = np.zeros([len(sentence), self.n_features])
        for i, w in enumerate(sentence):
            feat_matrix[i, self.get(w)] = 1
        assert feat_matrix.shape == (len(sentence), self.n_features)
        return feat_matrix

class BagOfWords(OneHot):
    def __init__(self, vocab_indices_path, max_vocab_size, argument, **kwargs):
        super().__init__(vocab_indices_path, max_vocab_size, argument, **kwargs)

    def extract_features(self, sentence):
        feats = np.sum(super().extract_features(sentence), axis=0, keepdims=True)
        assert feats.shape == (1, self.n_features)
        return feats
