from logging import getLogger
from datasets import Dataset
from collections import Counter

class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}
        for user in self.all_item_seqs:
            # import ipdb; ipdb.set_trace() # self.all_item_seqs[user] ['1881509818', 'B0048KGFHU', 'B0081JJVUC', 'B000N8OIE8', 'B004Y27DVY', 'B00D7ONGFC', 'B000NJY1YO', 'B00162ULZ0']
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])
        self.thr_cold_test_split(datasets)  # 增加字段 cold_test warm_test

        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])  ##  这里用Dataset包装了
    
        return datasets

    def cold_test_split(self,datasets:dict):
        train_dict = datasets["train"]
        test_dict  = datasets["test"]
        train_items = set()
        for seq in train_dict["item_seq"]:
            train_items.update(seq)
        
        add_dataset = {'warm_test': {'user': [], 'item_seq': []},
                       'cold_test': {'user': [], 'item_seq': []}}
        # 2) 测试集中每条序列的 target（默认取最后一个）
        for i in range(len(test_dict["user"])):
            user = test_dict["user"][i]
            target_item = test_dict["item_seq"][i][-1]
            if target_item in train_items:
                add_dataset['warm_test']['user'].append(user)
                add_dataset['warm_test']['item_seq'].append(test_dict["item_seq"][i])
            else:
                add_dataset['cold_test']['user'].append(user)
                add_dataset['cold_test']['item_seq'].append(test_dict["item_seq"][i])
        datasets.update(add_dataset)
        return 
    

    def thr_cold_test_split(self, datasets: dict, threshold: int = 5):
        train_dict = datasets["train"]
        test_dict  = datasets["test"]
    
        # 1) 统计训练集中每个 item 的出现次数（任何位置都算）
        train_item_cnt = Counter()
        for seq in train_dict["item_seq"]: # 94762
            train_item_cnt.update(seq)

        train_items_set = set()
        for seq in train_dict["item_seq"]:
            train_items_set.update(seq) # 25527
        add_dataset = {
            'warm_test': {'user': [], 'item_seq': []},
            'cold_test': {'user': [], 'item_seq': []}
        }
        import ipdb; ipdb.set_trace()
        # 2) 按 test 每条序列的 target（最后一个）划分 warm/cold
        # import json
        # sim_items_path = "/new_disk1/chenglei_shen/projects/GenRec-main/Edit/cold_test_sim_items.json"
        # with open(sim_items_path, "r") as f:
        #     sim_items_list = json.load(f)
        for i in range(len(test_dict["user"])):
            user = test_dict["user"][i]
            seq = test_dict["item_seq"][i]
            if len(seq) == 0:
                continue  # 或者按你的需求把空序列单独处理
            target_item = seq[-1]
            if train_item_cnt.get(target_item, 0) < threshold:
                add_dataset['cold_test']['user'].append(user)
                add_dataset['cold_test']['item_seq'].append(seq)
            else:# 交互次数大 
                # if target_item in set(sim_items_list): #如果target和cold item相似
                #     continue
                add_dataset['warm_test']['user'].append(user)
                add_dataset['warm_test']['item_seq'].append(seq)
        datasets.update(add_dataset)
        return


    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None: #### 关键，如果使用timestamp划分 就不进行后边处理
            dataset = self.split_data
            training_item = set()
            # for split in ['train', 'valid', 'test']: # train 736827 条
            ds = dataset['train'] # .sort('timestamp').map(lambda t: {'timestamp': int(t['timestamp'])})
            maxlen = 0
            # import ipdb; ipdb.set_trace()
            for user_id, seq in zip(ds['user'],ds['item_seq']): 
                training_item.add(seq[-1]) #22976
                if len(seq) > maxlen:
                    maxlen = len(seq)
                # training_item.update(history.split(' ')) # 22757  #
                # training_item.update([item_id]) # 22977
            print("maxlen:", maxlen)
            self.split_data['warm_test'] = self.split_data['test'].filter(lambda t: t['item_seq'][-1] in training_item)
            self.split_data['cold_test'] = self.split_data['test'].filter(lambda t: t['item_seq'][-1] not in  training_item)
            # import ipdb; ipdb.set_trace()
            return self.split_data


        split_strategy = self.config['split']
        if split_strategy in ['leave_one_out', 'last_out']:
            datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        self.split_data = datasets
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)