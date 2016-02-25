import numpy as np
from .misc_utils import get_logger

logger = get_logger(__name__)

class Extractor():
    """
    Base class with expected functions for any Extractor subclass.
    """
    def __init__(self, **kwargs):
        self.n_features = None

    def extract_features(self, sentences):
        raise NotImplementedError("This must be subclassed.")

class EmptyData():
    def __init__(self):
        self.vocab = {}
        self.vector_size = 300


class Word2Vec(Extractor):
    """
    This loads Word2Vec vectors from binary file format, as created
    by the Google News corpus. You should download that first.
    """
    def __init__(self, path, **kwargs):
        """
        Loads word vectors for words in vocab
        """
        self.path = path
        if 'random_vectors_only' in kwargs and kwargs['random_vectors_only']:
            self.data = EmptyData()
        else:
            self.data = self._load_from_binary_format(self.path)

        self.embedding_size = self.data.vector_size
        self.n_embeddings = len(self.data.vocab)
        self.n_features = self.embedding_size
        self.oov = np.random.randn(self.embedding_size)

    def extract_features(self, sentence):
        """
        It is constrained to a specific number of words per sentence.
        If the sentence is too long, we cut it off at the end.
        If it is too short, we add zero vectors to it.
        Returns features according to:
            words x embedding_size
        """
        feats = np.array([self[w] for w in sentence])
        assert feats.shape == (len(sentence), self.embedding_size)
        return feats


    def _load_from_binary_format(self, path):
        logger.info("Loading Word2Vec matrix, this will take a while...")
        import gensim
        data = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        logger.debug("Word2Vec matrix loaded")
        return data

    def __getitem__(self, s):
        if s in self.data.vocab:
            return self.data[s]
        else:
            return self.oov
