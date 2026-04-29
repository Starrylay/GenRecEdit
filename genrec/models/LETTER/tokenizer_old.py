import collections
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.models.LETTER.rqvae import RQVAE
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import list_to_str


class EmbDataset(torch.utils.data.Dataset):

    def __init__(self, embeddings: torch.Tensor):
        self.embeddings = embeddings
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)


class LETTERTokenizer(AbstractTokenizer):

    def __init__(self, config: dict, dataset: AbstractDataset):
        super(LETTERTokenizer, self).__init__(config, dataset)

        # TODO: check
        self.labels = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": []}

        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.base_user_token = sum(self.codebook_sizes) + 1
        self.n_user_tokens = self.config['n_user_tokens']
        self.eos_token = self.base_user_token + self.n_user_tokens
        self.ignored_label = -100

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(
            self.config['device'])

        meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
        for i in range(1, dataset.n_items):
            meta_sentences.append(
                dataset.item2meta[dataset.id_mapping['id2item'][i]])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'])

        sent_embs.tofile(output_path)
        return sent_embs

    @torch.no_grad()
    def _valid_collision(self, model: RQVAE, valid_data):
        model.eval()

        # iter_data = tqdm(
        #     valid_data,
        #     total=len(valid_data),
        #     ncols=100,
        #     desc="Evaluate   ",
        # )
        indices_set = set()

        num_sample = 0
        embs = [
            layer.embedding.weight.cpu().detach().numpy()
            for layer in model.rq.vq_layers
        ]

        # for idx, emb in enumerate(embs):
        #     centers, labels = self.constrained_km(emb)
        #     self.labels[str(idx)] = labels

        all_labels = None
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            labels = torch.LongTensor(labels).to(
                self.config['device']).unsqueeze(0)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

        # for batch_idx, data in enumerate(iter_data):
        for batch_idx, data in enumerate(valid_data):
            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.config['device'])
            # indices = model.get_indices(data, self.labels)
            indices = model.get_indices(data, all_labels)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / num_sample

        return collision_rate

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained

        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(n_clusters=n_clusters,
                                size_min=size_min,
                                size_max=n_clusters * 6,
                                max_iter=10,
                                n_init=10,
                                n_jobs=10,
                                verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        # t_labels = torch.from_numpy(clf.labels_)

        return t_centers, t_labels

    def vq_init(self, model: RQVAE, sent_embs: torch.Tensor, device):
        model.eval()
        original_data = EmbDataset(sent_embs.cpu())
        init_loader = DataLoader(original_data,
                                 num_workers=4,
                                 batch_size=len(original_data),
                                 shuffle=True,
                                 pin_memory=True)
        # print(len(init_loader))

        self.log("Initializing VQ with {} samples".format(len(init_loader)))
        iter_data = tqdm(
            init_loader,
            total=len(init_loader),
            ncols=100,
            desc="Initialization of vq",
        )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(device)

            model.vq_initialization(data)

    def _train_rqvae_epoch(self, model: RQVAE, train_data, epoch_idx,
                           optimizer: torch.optim.Optimizer):
        model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        # print(len(train_data))

        # iter_data = tqdm(
        #     train_data,
        #     total=len(train_data),
        #     ncols=100,
        #     desc=f"Train {epoch_idx}",
        # )
        embs = [
            layer.embedding.weight.cpu().detach().numpy()
            for layer in model.rq.vq_layers
        ]

        # for idx, emb in enumerate(embs):
        #     centers, labels = self.constrained_km(emb)
        #     self.labels[str(idx)] = labels

        all_labels = None
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            labels = torch.LongTensor(labels).to(
                self.config['device']).unsqueeze(0)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

        # for batch_idx, data in enumerate(iter_data):
        for batch_idx, data in enumerate(train_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.config['device'])
            optimizer.zero_grad()
            # out, rq_loss, indices, dense_out = model(data, self.labels)
            out, rq_loss, indices, dense_out = model(data, all_labels)

            loss, cf_loss, loss_recon, quant_loss = model.compute_loss(
                out, rq_loss, emb_idx, dense_out, xs=data)

            if torch.isnan(loss):
                raise ValueError("Training loss is nan")

            loss.backward()
            optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_cf_loss, quant_loss.item()

    def _train_rqvae(self, sent_embs: torch.Tensor, cf_emb: torch.Tensor,
                     model_path: str):

        device = self.config['device']

        # Initialize RQ-VAE model
        all_hidden_sizes = [sent_embs.shape[1]
                            ] + self.config['rqvae_hidden_sizes']

        data = EmbDataset(sent_embs.cpu())

        rqvae_model = RQVAE(
            in_dim=data.dim,
            num_emb_list=self.codebook_sizes,
            # e_dim=all_hidden_sizes[-1],
            e_dim=cf_emb.shape[-1],
            layers=self.config['rqvae_hidden_sizes'],
            dropout_prob=self.config['rqvae_dropout'],
            bn=self.config['rqvae_bn'],
            loss_type='mse',
            quant_loss_weight=self.config['rqvae_quant_weight'],
            kmeans_init=self.config['kmeans_init'],
            kmeans_iters=self.config['kmeans_iters'],
            sk_epsilons=[0, 0, 0, self.config['rqvae_sk_epsilon']],
            sk_iters=self.config['rqvae_sk_iters'],
            beta=self.config['rqvae_beta'],
            alpha=self.config['rqvae_alpha'],
            n_clusters=self.config['n_clusters'],
            sample_strategy=self.config['sample_strategy'],
            cf_embedding=cf_emb).to(device)

        self.log(rqvae_model)

        # TODO: debug先注释掉
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path))
            return rqvae_model

        batch_size = self.config['ravae_batch_size']
        num_epochs = self.config['rqvae_epoch']
        verbose = self.config['rqvae_verbose']

        optimizer = torch.optim.AdamW(rqvae_model.parameters(),
                                      lr=self.config['rqvae_lr'])
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)

        # labels = {"0": [], "1": [], "2": [], "3": [], "4": [], "5": []}

        best_loss = float('inf')
        best_collision_rate = float('inf')

        self.log("[TOKENIZER] Training RQ-VAE model...")

        # TODO: 运行费时间debug先去掉 正式运行加回来
        self.vq_init(rqvae_model, sent_embs, device)

        rqvae_model.train()
        for epoch in tqdm(range(num_epochs)):
            train_loss, train_recon_loss, cf_loss, quant_loss = self._train_rqvae_epoch(
                rqvae_model, dataloader, epoch, optimizer)

            if train_loss < best_loss:
                best_loss = train_loss

            if (epoch + 1) % verbose == 0:
                collision_rate = self._valid_collision(rqvae_model, dataloader)

                if collision_rate < best_collision_rate:
                    best_collision_rate = collision_rate

                    torch.save(rqvae_model.state_dict(), model_path)

                self.log(
                    f"[TOKENIZER] RQ-VAE training\n"
                    f"\tEpoch [{epoch+1}/{num_epochs}]\n"
                    f"\t  Training loss: {train_loss/ len(dataloader)}\n"
                    f"\t  Recosntruction loss: {train_recon_loss/ len(dataloader)}\n"
                    f"\t  Quantization loss: {quant_loss/ len(dataloader)}\n"
                    f"\t  CF loss: {cf_loss/ len(dataloader)}\n"

                    # f"\t  Training loss: {total_loss/ len(dataloader)}\n"
                    # f"\t  Unused codebook:{total_count/ len(dataloader)}\n"
                    # f"\t  Recosntruction loss: {total_rec_loss/ len(dataloader)}\n"
                    # f"\t  Quantization loss: {total_quant_loss/ len(dataloader)}\n"
                    f"\t  Collision rate: {collision_rate}\n")

        self.log("[TOKENIZER] RQ-VAE training complete.")

        return rqvae_model

    def _check_collision(self, str_sem_ids):
        tot_item = len(str_sem_ids)
        tot_ids = len(set(str_sem_ids.tolist()))
        self.log(
            f'[TOKENIZER] Collision rate: {(tot_item - tot_ids) / tot_item}')
        return tot_item == tot_ids

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(
            f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}'
        )
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """

        # sem_ids_path = os.path.join(
        #     dataset.cache_dir, 'processed',
        #     f'{os.path.basename(self.config["sent_emb_model"])}_RQ(LETTER){list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
        # )
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_RQVAE(LETTER){list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            # Load or encode sentence embeddings
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb')
            if os.path.exists(sent_emb_path):
                self.log(
                    f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...'
                )
                sent_embs = np.fromfile(sent_emb_path,
                                        dtype=np.float32).reshape(
                                            -1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            # PCA
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'],
                          whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(
                f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)

            self.log(
                f'[TOKENIZER] Semantic IDs not found. Training RQ-VAE model...'
            )

            cf_emb = torch.load(self.config['CF_emb_path'])
            self.log(f'[TOKENIZER] CF embedding shape: {cf_emb.shape}')
            # assert cf_emb.shape[0] == len(sent_embs)

            cf_emb = cf_emb[1:dataset.n_items]

            # cf_emb_training = cf_emb[training_item_mask].to(
            #     self.config['device'])
            cf_emb_training = cf_emb[training_item_mask].detach().cpu().numpy()

            embs_for_training = torch.FloatTensor(
                sent_embs[training_item_mask]).to(self.config['device'])
            sent_embs = torch.FloatTensor(sent_embs).to(self.config['device'])
            # model_path = os.path.join(dataset.cache_dir, 'processed/rqvae.pth')
            model_path = os.path.join(dataset.cache_dir,
                                      'processed/rqvae_letter.pth')
            rqvae_model = self._train_rqvae(embs_for_training, cf_emb_training,
                                            model_path)
            rqvae_model.load_state_dict(
                torch.load(model_path, map_location=self.config['device']))

            self._generate_semantic_id(rqvae_model, sent_embs, sem_ids_path)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _generate_semantic_id(self, model: RQVAE, sent_embs: torch.Tensor,
                              sem_ids_path: str) -> None:
        model.eval()

        batch_size = self.config['ravae_batch_size']
        data = EmbDataset(sent_embs.cpu())
        dataloader = DataLoader(data,
                                num_workers=4,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True)

        all_indices = []
        all_indices_str = []
        # prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>"]

        embs = [
            layer.embedding.weight.cpu().detach().numpy()
            for layer in model.rq.vq_layers
        ]

        # labels = {"0": [], "1": [], "2": [], "3": []}
        # for idx, emb in enumerate(embs):
        #     centers, label = self.constrained_km(emb)
        #     labels[str(idx)] = label

        all_labels = None
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            labels = torch.LongTensor(labels).to(
                self.config['device']).unsqueeze(0)

            if all_labels is None:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

        for d in tqdm(dataloader):
            d, emb_idx = d[0], d[1]
            d = d.to(self.config['device'])

            # indices = model.get_indices(d, use_sk=False)
            # indices = model.get_indices(d, labels, use_sk=False)
            indices = model.get_indices(d, all_labels, use_sk=False)

            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = index.tolist()
                # for i, ind in enumerate(index):
                #     code.append(prefix[i].format(int(ind)))

                # for i, ind in enumerate(index):
                #     code.append(int(ind))

                all_indices.append(code)
                # all_indices_str.append(str(code))
                all_indices_str.append('-'.join([str(x) for x in code]))
            # break

        all_indices = np.array(all_indices)
        all_indices_str = np.array(all_indices_str)

        for vq in model.rq.vq_layers[:-1]:
            vq.sk_epsilon = 0.0

        if model.rq.vq_layers[-1].sk_epsilon == 0.0:
            model.rq.vq_layers[-1].sk_epsilon = 0.003

        # TODO: debug
        for _ in range(30):
            # for _ in range(1):
            if self._check_collision(all_indices_str):
                break

            collision_item_groups = self._get_collision_items(all_indices_str)

            for collision_items in collision_item_groups:
                d = data[collision_items]
                d = d[0].to(self.config['device'])
                # indices = model.get_indices(d, labels, use_sk=True)
                indices = model.get_indices(d, all_labels, use_sk=True)

                indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                for item, index in zip(collision_items, indices):
                    code = index.tolist()
                    # for i, ind in enumerate(index):
                    #     code.append(prefix[i].format(int(ind)))

                    # for i, ind in enumerate(index):
                    #     code.append(int(ind))

                    all_indices[item] = code
                    all_indices_str[item] = '-'.join([str(x) for x in code])

        # self.log("All indices number: ", len(all_indices))
        self.log(f"All indices number: {len(all_indices)}")

        indices_count = defaultdict(int)
        for index in all_indices_str:
            indices_count[index] += 1

        # self.log("Max number of conflicts: ", max(indices_count.values()))
        self.log(f"Max number of conflicts: {max(indices_count.values())}")

        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str.tolist()))
        self.log("Collision Rate :{}".format(
            (tot_item - tot_indice) / tot_item))

        item2sem_ids = {}
        for item_id, indices in enumerate(all_indices.tolist()):
            item = self.id2item[int(item_id) + 1]
            item2sem_ids[item] = list(indices)

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            # json.dump(all_indices_str, f)
            json.dump(item2sem_ids, f)

    def _get_collision_items(self, str_sem_ids):
        sem_id2item = defaultdict(list)
        for i, str_sem_id in enumerate(str_sem_ids):
            sem_id2item[str_sem_id].append(i)

        collision_item_groups = []
        for str_sem_id in sem_id2item:
            if len(sem_id2item[str_sem_id]) > 1:
                collision_item_groups.append(sem_id2item[str_sem_id])

        return collision_item_groups

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        sem_id_offsets = [0]
        for digit in range(1, self.n_digit):
            sem_id_offsets.append(sem_id_offsets[-1] +
                                  self.codebook_sizes[digit - 1])
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += sem_id_offsets[digit] + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

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
