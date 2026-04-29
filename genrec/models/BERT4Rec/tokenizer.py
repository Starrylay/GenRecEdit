from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class BERT4RecTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(BERT4RecTokenizer, self).__init__(config, dataset)

        self.item2tokens = dataset.item2id
        self.eos_token = len(self.item2tokens) + 1
        self.ignored_label = -100

    def _init_tokenizer(self):
        pass

    def _tokenize_test(self, item_seq: list):
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2tokens[item_seq[-1]]]

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_train(self, item_seq: list):
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        return input_ids, attention_mask, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':

            # TODO: 训练最大序列长度和测试差了1个 可能导致最后一个位置pos emb没被训练过?
            # 所以训练的时候也应该取最大历史长度+1

            all_input_ids, all_attention_mask, all_seq_lens = [], [], []

            for i in range(2, len(item_seq) + 1):
                # cur_item_seq = item_seq[max(0, i - max_item_seq_len):i]
                cur_item_seq = item_seq[max(0, i - max_item_seq_len - 1):i]
                input_ids, attention_mask, seq_lens = self._tokenize_train(
                    cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_test(
                item_seq[-(max_item_seq_len + 1):])
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the datasets using the specified tokenizer function.

        Args:
            datasets (dict): A dictionary containing the datasets to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized datasets.
        """
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: ')

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len']
