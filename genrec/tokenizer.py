from logging import getLogger

import ipdb

from genrec.dataset import AbstractDataset


class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {'train': None, 'val': None, 'test': None}

    def _init_tokenizer(self):
        raise NotImplementedError('Tokenizer initialization not implemented.')

    def tokenize(self, datasets):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError('Maximum token sequence length not implemented.')

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
    
import numpy as np
class SeqRecTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super().__init__(config, dataset)

        self.item2tokens = dataset.item2id

    def _init_tokenizer(self):
        pass

    def _tokenize_items(self, item_seq: list) -> tuple:
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2tokens[item_seq[-1]]]

        # if 'B001HKPUKC' == item_seq[-1]:
        #     pass
        # import ipdb; ipdb.set_trace()
        # assert not np.any(labels == '0'), labels # No padding in labels
        return input_ids, attention_mask, labels, seq_lens
