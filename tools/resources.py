from . import pdtb_utils
from .misc_utils import get_config, get_en_model
from .pdtb_utils import get_relations
from collections import Counter
import numpy as np
from .misc_utils import get_logger

logger = get_logger(__name__)

class Resource():
    def __init__(self, path):
        self.path = path
        self.data = list(self._load_data(path))

    def _load_data(self, path):
        raise NotImplementedError("This class must be subclassed.")


class ConllRelations(Resource):
    def __init__(self, path):
        super(ConllRelations, self).__init__(path)

    def _load_data(self, path):
        return get_relations(self.path)

    def get_sentences(self):
        """
        This gives you the raw sentences.
        Useful if you want to do some vocab counting or similar.
        """
        en_model = get_en_model()
        if not hasattr(self, 'sentences'):
            for rel in self.data:
                yield [w.norm_ for w in en_model(rel.arg1_text())]
                yield [w.norm_ for w in en_model(rel.arg2_text())]
        else:
            return self.sentences

    def get_vocab(self):
        """
        Returns a frequency dictionary.
        """
        if not hasattr(self, 'vocab'):
            self.vocab = Counter((w for s in self.get_sentences() for w in s))
            return self.vocab
        else:
            return self.vocab
