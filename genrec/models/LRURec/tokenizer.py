from genrec.dataset import AbstractDataset
from genrec.tokenizer import SeqRecTokenizer


class LRURecTokenizer(SeqRecTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(LRURecTokenizer, self).__init__(config, dataset)
