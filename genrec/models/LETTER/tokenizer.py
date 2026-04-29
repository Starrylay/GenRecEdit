import importlib

import yaml

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class LETTERTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(LETTERTokenizer, self).__init__(config, dataset)

        tokenizer_name = self.config['tokenizer_name']
        try:
            tokenizer_class = getattr(
                importlib.import_module(
                    f'genrec.tokenizers.{tokenizer_name}.tokenizer'),
                f'{tokenizer_name}Tokenizer')
        except:
            raise ValueError(f'Tokenizer "{tokenizer_name}" not found.')

        self.log(f'[TOKENIZER] Loading tokenizer config')
        tokenizer_config: dict = yaml.safe_load(
            open(f'genrec/tokenizers/{tokenizer_name}/config.yaml', 'r'))

        for key in tokenizer_config.keys():
            if key in config.keys():
                tokenizer_config[key] = config[key]
            self.log(f"{key}: {tokenizer_config[key]}")
        config.update(tokenizer_config)
        self.config = config

        self.tokenizer = tokenizer_class(config, dataset)

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        # self.item2id = dataset.item2id
        # self.item2tokens = self._init_tokenizer(dataset)
        self.item2tokens = self.tokenizer.item2tokens

        self.base_user_token = sum(self.codebook_sizes) + 1
        self.n_user_tokens = self.config['n_user_tokens']
        self.eos_token = self.base_user_token + self.n_user_tokens
        self.ignored_label = -100

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        """
        return self.config['rq_n_codebooks']

    @property
    def codebook_sizes(self):
        """
        Returns the codebook size for the LCRec tokenizer.

        If `rq_codebook_size` is a list, it returns the list as is.
        If `rq_codebook_size` is an integer, it returns a list with `n_digit` elements,
        where each element is equal to `rq_codebook_size`.

        Returns:
            list: The codebook size for the LCRec tokenizer.
        """
        if isinstance(self.config['rq_codebook_size'], list):
            return self.config['rq_codebook_size']
        else:
            return [self.config['rq_codebook_size']] * self.n_digit

    def _token_single_user(self, user: str) -> int:
        """
        Tokenizes a single user.

        Args:
            user (str): The user to tokenize.

        Returns:
            int: The tokenized user ID.

        """
        user_id = self.user2id[user]
        return self.base_user_token + user_id % self.n_user_tokens

    def _token_single_item(self, item: str) -> int:
        """
        Tokenizes a single item.

        Args:
            item (str): The item to be tokenized.

        Returns:
            list: The tokens corresponding to the item.
        """
        return self.item2tokens[item]

    def _tokenize_once(self, example: dict) -> tuple:
        """
        Tokenizes a single example.

        Args:
            example (dict): A dictionary containing the example data.

        Returns:
            tuple: A tuple containing the tokenized input_ids, attention_mask, and labels.
        """
        max_item_seq_len = self.config['max_item_seq_len']

        # input_ids
        user_token = self._token_single_user(example['user'])
        input_ids = [user_token]
        for item in example['item_seq'][:-1][-max_item_seq_len:]:
            input_ids.extend(self._token_single_item(item))
        input_ids.append(self.eos_token)
        input_ids.extend([self.padding_token] *
                         (self.max_token_seq_len - len(input_ids)))

        # attention_mask
        item_seq_len = min(len(example['item_seq'][:-1]), max_item_seq_len)
        attention_mask = [1] * (self.n_digit * item_seq_len + 2)
        attention_mask.extend([0] *
                              (self.max_token_seq_len - len(attention_mask)))

        # labels
        labels = list(self._token_single_item(
            example['item_seq'][-1])) + [self.eos_token]

        return input_ids, attention_mask, labels

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.

        Args:
            example (dict): The input example containing user and item sequence.
            split (str): The split type, either 'train' or any other value.

        Returns:
            dict: A dictionary containing the tokenized input, attention mask, and labels.
                - If split is 'train', returns:
                    {
                        'input_ids': List[List[int]],
                        'attention_mask': List[List[int]],
                        'labels': List[List[int]]
                    }
                - If split is not 'train', returns:
                    {
                        'input_ids': List[int],
                        'attention_mask': List[int],
                        'labels': List[int]
                    }
        """
        if split == 'train':
            n_return_examples = len(example['item_seq'][0]) - 1
            all_input_ids, all_attention_mask, all_labels = [], [], []
            for i in range(n_return_examples):
                cur_example = {
                    'user': example['user'][0],
                    'item_seq': example['item_seq'][0][:i + 2]
                }
                input_ids, attention_mask, labels = self._tokenize_once(
                    cur_example)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels
            }
        else:
            input_ids, attention_mask, labels = self._tokenize_once({
                k: v[0]
                for k, v in example.items()
            })
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels]
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.

        Returns:
            dict: A dictionary of tokenized datasets.
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
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length for the TIGER tokenizer.
        """
        # +2 for user token and eos token
        return self.config['max_item_seq_len'] * self.n_digit + 2
