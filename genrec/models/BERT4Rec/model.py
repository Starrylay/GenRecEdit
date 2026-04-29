import torch
from transformers import BertConfig, BertForMaskedLM

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class BERT4Rec(AbstractModel):

    # TODO: Implement BERT4Rec model
    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(BERT4Rec, self).__init__(config, dataset, tokenizer)

        bertconfig = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size'],
            hidden_act=config['hidden_act'],
            hidden_dropout_prob=config['hidden_dropout_prob'],
            attention_probs_dropout_prob=config[
                'attention_probs_dropout_prob'],
            # max_position_embeddings=tokenizer.max_token_seq_len,
            max_position_embeddings=tokenizer.max_token_seq_len +
            1,  # +1 for next item
            initializer_range=config['initializer_range'],
            layer_norm_eps=config['layer_norm_eps'],
            pad_token_id=tokenizer.padding_token,
        )
        self.mask_token_id = tokenizer.eos_token  # TODO: check this
        self.mask_ratio = config['mask_ratio']

        self.bert = BertForMaskedLM(bertconfig)
        self.loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.ignored_label, reduction='none')

    @property
    def n_parameters(self) -> str:
        """
        Get the number of parameters in the model.

        Returns:
            str: A string representation of the number of parameters in the model.
        """
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        emb_params = sum(
            p.numel() for p in self.bert.get_input_embeddings().parameters()
            if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the logits and the loss.

        Args:
            batch (dict): The input batch.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        noisy_batch, masked_indices = self.get_masked_seqs(
            input_ids, attention_mask)

        outputs = self.bert(input_ids=noisy_batch,
                            attention_mask=attention_mask)
        logits = outputs.logits

        input_ids = input_ids.masked_fill(~(attention_mask.bool()),
                                          self.tokenizer.ignored_label)

        loss = self.loss_fct(logits[masked_indices], input_ids[masked_indices])
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        outputs.loss = loss

        return outputs

    def get_masked_seqs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        b, l = input_ids.shape

        masked_indices = torch.rand((b, l),
                                    device=input_ids.device) < self.mask_ratio
        noisy_batch = torch.where(masked_indices, self.mask_token_id,
                                  input_ids)

        noisy_batch = noisy_batch.masked_fill(~(attention_mask.bool()), 0)
        masked_indices = masked_indices.masked_fill(~(attention_mask.bool()),
                                                    False)

        return noisy_batch, masked_indices

    def generate(self, batch, n_return_sequences=1):
        batch_size, seq_len = batch['input_ids'].shape

        input_ids = torch.ones((batch_size, seq_len + 1),
                               device=self.bert.device,
                               dtype=torch.long) * self.mask_token_id
        attention_mask = torch.ones((batch_size, seq_len + 1),
                                    device=self.bert.device,
                                    dtype=torch.long)

        input_ids[:, :seq_len] = batch['input_ids']
        attention_mask[:, :seq_len] = batch['attention_mask']

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -1]
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)
