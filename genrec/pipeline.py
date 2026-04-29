from logging import getLogger
from typing import Union
import torch
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datetime import datetime
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import get_config, init_seed, init_logger, init_device, \
    get_dataset, get_tokenizer, get_model, get_trainer, log

from datasets import load_from_disk
from datasets import concatenate_datasets

class Pipeline:
    def __init__(
        self,
        model_name: Union[str, AbstractModel],
        dataset_name: Union[str, AbstractDataset],
        tokenizer: AbstractTokenizer = None,
        trainer = None,
        config_dict: dict = None,
        config_file: str = None,
    ):
        self.config = get_config(
            model_name=model_name,
            dataset_name=dataset_name,
            config_file=config_file,
            config_dict=config_dict  # 从命令行来的变量，可以直接在命令行中定义，后文用config[""]取值 ,命令刚会覆盖其他定义     
            )
 
        # import ipdb; ipdb.set_trace()
        # Automatically set devices and ddp

        self.config['device'], self.config['use_ddp'] = init_device() 
        # print(self.config)
        # Accelerator
        self.project_dir = os.path.join(
            self.config['tensorboard_log_dir'],
            self.config["dataset"],
            self.config["model"]
        )
        self.accelerator = Accelerator(log_with='tensorboard', project_dir=self.project_dir)
        self.config['accelerator'] = self.accelerator
        
        # Seed and Logger
        init_seed(self.config['rand_seed'], self.config['reproducibility'])
        init_logger(self.config)
        self.logger = getLogger()
        self.log(f'{self.config}')
        self.log(f'Device: {self.config["device"]}')

        # Dataset
        # import ipdb; ipdb.set_trace()
        self.raw_dataset = get_dataset(dataset_name)(self.config)
        # import ipdb; ipdb.set_trace()
        self.log(self.raw_dataset)
        self.split_datasets = self.raw_dataset.split()   # 这里把数据集划分好了 返回的是字典 key是'train','val','test' value是Dataset对象

        # datasets = {'train': {'user': [], 'item_seq': []},
        #     'val': {'user': [], 'item_seq': []},
        #     'test': {'user': [], 'item_seq': []}}
        #     "warm_test": {'user': [], 'item_seq': []},
        #     'cold_test': {'user': [], 'item_seq': []}}

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer(self.config, self.raw_dataset)
        else:
            assert isinstance(model_name, str), 'Tokenizer must be provided if model_name is not a string.'
            self.tokenizer = get_tokenizer(model_name)(self.config, self.raw_dataset)  # 这里训练了RQVAE tokenizer 的init
        # import ipdb; ipdb.set_trace()
        if self.config.get('pretrained_model_path', None) is not None:
            retain = {'cold_test', 'warm_test', 'test'}
            self.split_datasets = {k: v for k, v in self.split_datasets.items() if k in retain}

        self.tokenized_datasets = self.tokenizer.tokenize(self.split_datasets)  ## 这里把数据集都tokenize了（变成了sids）

        if self.config.get('retrain',False) == True:
            out_path=f"/new_disk1/chenglei_shen/projects/GenRec-main/Edit/{self.config['category']}/edit_requests_cold_test_augmented_5_train.json"
            datasets_cold_test_augmented_5_train = load_from_disk(out_path)
            # 合并训练集
            self.log( f'Before Augmentation, train dataset size: {len(self.tokenized_datasets["train"])}')
            self.tokenized_datasets['train'] = concatenate_datasets([
                self.tokenized_datasets["train"],
                datasets_cold_test_augmented_5_train,
            ])
            self.log( f'After Augmentation, train dataset size: {len(self.tokenized_datasets["train"])}')

        if self.config.get('fintune',False) == True:
            out_path=f"/new_disk1/chenglei_shen/projects/GenRec-main/Edit/{self.config['category']}/edit_requests_cold_test_augmented_5_train.json"
            datasets_cold_test_augmented_5_train = load_from_disk(out_path)
            # 替代训练集
            self.log( f'Before Augmentation, train dataset size: {len(self.tokenized_datasets["train"])}')
            self.tokenized_datasets['train'] = datasets_cold_test_augmented_5_train
            # 融入一部分原训练集
            # num_original = int(0.01 * len(self.tokenized_datasets["train"]))
            # self.tokenized_datasets['train'] = concatenate_datasets([
            #     self.tokenized_datasets["train"].select(range(num_original)),
            #     datasets_cold_test_augmented_5_train,
            # ])

            self.log( f'After Augmentation, train dataset size: {len(self.tokenized_datasets["train"])}')           
            

        # import ipdb; ipdb.set_trace()
        # Model
        with self.accelerator.main_process_first():
            self.model = get_model(model_name)(self.config, self.raw_dataset, self.tokenizer)
            # import ipdb; ipdb.set_trace()

            @torch.no_grad()
            def param_stats(model, name_hint=None):
                for name, p in model.named_parameters():
                    if p is None: 
                        continue
                    if (name_hint is None) or (name_hint in name):
                        x = p.detach().float().cpu()
                        print(f"{name:60s} shape={tuple(p.shape)} "
                            f"min={x.min():.6f} max={x.max():.6f} mean={x.mean():.6f} std={x.std():.6f}")
                        break
            param_stats(self.model)                 # 随便挑第一个参数
            if self.config.get('fintune',False) == True:
                print("fintune from base model:", self.config['fintune_model_path_base'])
                # self.model = self.accelerator.unwrap_model(self.model)
                self.model.load_state_dict(torch.load(self.config['fintune_model_path_base']))
                param_stats(self.model)


        # self.log(self.model)
        self.log(self.model.n_parameters)
        # import ipdb; ipdb.set_trace()
        # Trainer
        if trainer is not None:
            self.trainer = trainer
        else:
            self.trainer = get_trainer(model_name)(self.config, self.model, self.tokenizer)

    def run(self):
        # DataLoader
        if self.config.get('pretrained_model_path', None) is None:
            train_dataloader = DataLoader(
                self.tokenized_datasets['train'],
                batch_size=self.config['train_batch_size'],
                shuffle=True,
                collate_fn=self.tokenizer.collate_fn['train']
            )
            # import ipdb; ipdb.set_trace()
            val_dataloader = DataLoader(
                self.tokenized_datasets['val'],
                batch_size=self.config['eval_batch_size'],
                shuffle=False,
                collate_fn=self.tokenizer.collate_fn['val']
            )

        test_dataloader = DataLoader(
            self.tokenized_datasets['test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )
        cold_test_dataloader = DataLoader(
            self.tokenized_datasets['cold_test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )
        # import  ipdb; ipdb.set_trace()
        warm_test_dataloader = DataLoader(
            self.tokenized_datasets['warm_test'],
            batch_size=self.config['eval_batch_size'],
            shuffle=False,
            collate_fn=self.tokenizer.collate_fn['test']
        )

        # 如果存在pth文件，则加载模型直接测试
        # import ipdb; ipdb.set_trace()
        start = datetime.now()
        if self.config.get('pretrained_model_path', None) is None or self.config.get('fintune',False) == True: # 重训 or fintune
            self.trainer.fit(train_dataloader, val_dataloader)
            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)
            self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
            if self.accelerator.is_main_process:
                self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')
            end = datetime.now()

            self.log("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("The total time is ", (end-start).total_seconds()/60)
            self.log("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.log("+++++++++++++++++++++++++++++++++++++++++++++++++++")

        
        # self.trainer.fit(train_dataloader, val_dataloader)
        #             self.accelerator.wait_for_everyone()
        #             self.model = self.accelerator.unwrap_model(self.model)
        #             self.model.load_state_dict(torch.load(self.trainer.saved_model_ckpt))
        #             if self.accelerator.is_main_process:
        #                 self.log(f'Loaded best model checkpoint from {self.trainer.saved_model_ckpt}')

        else:
            self.model.load_state_dict(torch.load(self.config['pretrained_model_path']))
            if self.accelerator.is_main_process:
                self.log(f'Loaded best model checkpoint from {self.config["pretrained_model_path"]}')

    
        self.model, test_dataloader,cold_test_dataloader,warm_test_dataloader = self.accelerator.prepare(
            self.model, test_dataloader, cold_test_dataloader,warm_test_dataloader)
        # )
        # self.model, cold_test_dataloader = self.accelerator.prepare(
        #     self.model, cold_test_dataloader
        # )
        # self.model, warm_test_dataloader = self.accelerator.prepare(
        #     self.model, warm_test_dataloader
        # )

        test_results = self.trainer.evaluate(test_dataloader)
        cold_test_results = self.trainer.evaluate(cold_test_dataloader, split='cold_test')
        warm_test_results = self.trainer.evaluate(warm_test_dataloader, split='warm_test')
        
        if self.accelerator.is_main_process:
            for key in test_results:
                self.accelerator.log({f'Test_Metric/{key}': test_results[key]})
        self.log(f'Test Results: {test_results}')
        if self.accelerator.is_main_process:
            for key in cold_test_results:
                self.accelerator.log({f'Cold_Test_Metric/{key}': cold_test_results[key]})
        self.log(f'Cold Test Results: {cold_test_results}')

        if self.accelerator.is_main_process:
            for key in warm_test_results:
                self.accelerator.log({f'Warm_Test_Metric/{key}': warm_test_results[key]})
        self.log(f'Warm Test Results: {warm_test_results}')

        self.trainer.end()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
