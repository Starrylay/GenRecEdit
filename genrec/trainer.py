import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler
from accelerate import Accelerator

from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.evaluator import Evaluator
from genrec.utils import get_file_name, get_total_steps, config_for_log, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],self.config["model"]+"_"+self.config['category'],
            get_file_name(self.config, suffix='.pth')
        )
        # import ipdb; ipdb.set_trace()
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1
        # import ipdb; ipdb.set_trace()
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                # import ipdb; ipdb.set_trace()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # Evaluation
            if (epoch + 1) % self.config[''] == 0:
                all_results = self.ealuate(val_dataloader, split='val')
                if self.acceleratorv.is_main_process:
                    for key in all_results:
                        self.accelerator.leval_intervalog({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                # if val_score > best_val_score:
                best_val_score = val_score
                best_epoch = epoch + 1
                if self.accelerator.is_main_process:
                    if self.config['use_ddp']: # unwrap model for saving
                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                    else:
                        torch.save(self.model.state_dict(), self.saved_model_ckpt)
                    self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    break
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )

        # 收集dataloader的所有labels
        labels_all_list = []
        if 'test' in split:
            labels_all_set=set()
            for batch in val_progress_bar:
                with torch.no_grad():
                    batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                    if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                        all_labels = self.accelerator.gather_for_metrics(batch['labels'])
                    else:
                        all_labels = batch['labels']
                    batch_labels = []
                    for i, label in enumerate(all_labels):
                        cur_label = label.detach().cpu().tolist()
                        cur_label = cur_label[:3]
                        # if self.evaluator.eos_token in cur_label:
                        #     eos_pos = cur_label.index(self.evaluator.eos_token)
                        #     cur_label = cur_label[:eos_pos]
                        # import ipdb; ipdb.set_trace()
                        batch_labels.append(tuple(cur_label))
                    labels_all_set.update(batch_labels)
            labels_all_list = [list(x) for x in labels_all_set]  
            # import ipdb; ipdb.set_trace()
            print("done")
            
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )

        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels,labels_all_list)
                else:
                    # import ipdb; ipdb.set_trace()
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    # import ipdb; ipdb.set_trace()
                    results = self.evaluator.calculate_metrics(preds, batch['labels'], labels_all_list)
                
                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for k in self.config['topk']:
            iid_key = f"iid_ratio@{k}"
            if iid_key in all_results:
                output_results[iid_key] = torch.cat(all_results[iid_key]).mean().item() / k

            ndcg_key = f"ndcg@{k}"
            output_results[ndcg_key] = torch.cat(all_results[ndcg_key]).mean().item()
        return output_results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
