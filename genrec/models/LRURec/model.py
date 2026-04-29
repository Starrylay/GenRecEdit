import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel, SeqRecOutput
from genrec.tokenizer import AbstractTokenizer


class LRURec(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(LRURec, self).__init__(config, dataset, tokenizer)

        self.n_layers = config['n_layers']
        self.d_model = config['d_model']
        self.dropout = config['dropout']

        self.item_embeddings = nn.Embedding(
            tokenizer.vocab_size,
            self.d_model,
            padding_idx=tokenizer.padding_token)

        self.lru_blocks = nn.ModuleList([
            LRUBlock(hidden_size=self.d_model, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])

        self.loss_fct = torch.nn.CrossEntropyLoss()

        # self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

            for n, p in self.named_parameters():
                if not 'layer_norm' in n and 'params_log' not in n:
                    if torch.is_complex(p):
                        p.real.uniform_(2 * l - 1, 2 * u - 1)
                        p.imag.uniform_(2 * l - 1, 2 * u - 1)
                        p.real.erfinv_()
                        p.imag.erfinv_()
                        p.real.mul_(std * math.sqrt(2.))
                        p.imag.mul_(std * math.sqrt(2.))
                        p.real.add_(mean)
                        p.imag.add_(mean)
                    else:
                        p.uniform_(2 * l - 1, 2 * u - 1)
                        p.erfinv_()
                        p.mul_(std * math.sqrt(2.))
                        p.add_(mean)

    def _forward(self, batch: dict) -> torch.Tensor:
        rec_his = batch['input_ids']
        rec_his_mask = torch.where(rec_his == 0, 0, 1).bool()

        rec_his_emb = self.item_embeddings(rec_his)

        # left padding to the power of 2
        seq_len = rec_his_emb.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        rec_his_emb = F.pad(rec_his_emb,
                            (0, 0, 2**log2_L - rec_his_emb.size(1), 0, 0, 0))
        mask_ = F.pad(rec_his_mask,
                      (2**log2_L - rec_his_mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        for lru_block in self.lru_blocks:
            rec_his_emb = lru_block.forward(rec_his_emb, mask_)
        rec_his_emb = rec_his_emb[:, -seq_len:]  # B x L x D (64)

        last_hidden = self.gather_index(rec_his_emb, batch['seq_lens'] - 1)

        logits = torch.matmul(last_hidden,
                              self.item_embeddings.weight.transpose(0, 1))

        return logits

    def forward(self, batch: dict) -> torch.Tensor:
        logits = self._forward(batch)
        labels = batch['labels'].view(-1)
        loss = self.loss_fct(logits, labels)

        return SeqRecOutput(logits=logits, loss=loss)

    def gather_index(self, output, index):
        """
        Gather the output at a specific index.

        Args:
            output: The output tensor.
            index: The index tensor.

        Returns:
            torch.Tensor: The gathered output.
        """
        index = index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        return output.gather(dim=1, index=index).squeeze(1)

    def generate(self, batch, n_return_sequences=1):
        logits = self._forward(batch)
        preds = logits.topk(n_return_sequences, dim=-1).indices
        return preds.unsqueeze(-1)


class LRUBlock(nn.Module):

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.lru_layer = LRULayer(d_model=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden_size,
                                                    d_ff=hidden_size * 4,
                                                    dropout=dropout)

    def forward(self, x, mask):
        x = self.lru_layer(x, mask)
        x = self.feed_forward(x)
        return x


class LRULayer(nn.Module):

    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 *
                                            (r_max**2 - r_min**2) + r_min**2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(
            torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda)**2))
        self.params_log = nn.Parameter(
            torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size,
                                 self.hidden_size,
                                 bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size,
                                  self.embed_size,
                                  bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()

        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2**i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu

        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        for i in range(log2_L):
            h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x)  # residual connection introduced above


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)
