from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class SASRecTokenizer(AbstractTokenizer):
    """
    Tokenizer for SASRec model.

    An example:
        0: padding
        1-n_items: item tokens
        n_items+1: eos token

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        item2tokens (dict): A dictionary mapping items to their internal IDs.
        eos_token (int): The end-of-sequence token.
        ignored_label (int): Should be -100. Used to ignore the loss for padding tokens in `transformers`.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        super(SASRecTokenizer, self).__init__(config, dataset)

        self.item2tokens = dataset.item2id
        self.eos_token = len(self.item2tokens) + 1
        self.ignored_label = -100

    def _init_tokenizer(self):
        pass

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.

        Args:
            item_seq (list): The item sequence that contains the first n items.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, labels, and seq_lens.
        """
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2tokens[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.

        Args:
            item_seq (list): The item sequence.

        Returns:
            tuple: A tuple containing the tokenized input IDs, attention mask, labels, and seq_lens.
        """
        input_ids = [self.item2tokens[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2tokens[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing the item sequence.
            split (str): The split type ('train' or 'val' or 'test').

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            # Tokenize the first n items if len(item_seq) <= max_item_seq_len + 1
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                # Add 1 as the target item is not included in the input sequence
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # Tokenize the later items if len(item_seq) > max_item_seq_len + 1
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
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
                desc=f'Tokenizing {split} set: '
            )

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
