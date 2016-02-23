import numpy as np
from .misc_utils import get_logger

logger = get_logger(__name__)

class Extractor():
    """
    Base class with expected functions for any Extractor subclass.
    """
    def __init__(self, **kwargs):
        pass

    def extract_features(self, sentences):
        raise NotImplementedError("This must be subclassed.")

class Word2Vec(Extractor):
    """
    This loads Word2Vec vectors from binary file format, as created
    by the Google News corpus. You should download that first.
    """
    def __init__(self, path, max_words_in_sentence, **kwargs):
        """
        Loads word vectors for words in vocab
        """
        self.path = path
        self.data = self._load_from_binary_format(self.path)
        self.embedding_size = self.data.vector_size
        self.n_embeddings = len(self.data.vocab)
        self.oov = self._create_oov_vector()
        self.max_words_in_sentence = max_words_in_sentence

    def _create_oov_vector(self):
        return np.random.randn(self.embedding_size)

    def __getitem__(self, s):
        if s in self.data.vocab:
            return self.data[s]
        else:
            return self.oov

    def extract_features(self, sentences):
        """
        It is constrained to a specific number of words per sentence.
        If the sentence is too long, we cut it off at the end.
        If it is too short, we add zero vectors to it.
        Returns features according to:
            sentences x words x embedding_size
        """
        feats = []
        for s in sentences:
            s = s[:self.max_words_in_sentence]
            fill_out = np.zeros([self.max_words_in_sentence - len(s), self.embedding_size])
            feat = np.array([self[w] for w in s])
            feat = np.concatenate([feat, fill_out], axis=0)
            feats.append(feat)

        concatenated = np.array(feats)
        logger.info("Features extracted")
        return concatenated

    def _load_from_binary_format(self, path):
        logger.info("Loading Word2Vec matrix, this will take a while...")
        import gensim
        data = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
        logger.debug("Word2Vec matrix loaded")
        return data
