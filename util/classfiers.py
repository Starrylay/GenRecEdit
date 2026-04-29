from torch import nn
import torch


class Logistic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d)
        self.act = nn.GELU()
        self.w1 = nn.Linear(d, 1)

    def forward(self, x):
        x = self.w1(self.act(self.w(x)))   # (N, 1)
        return x.squeeze(-1)               # (N,)

    @torch.no_grad()
    def acc(self, X, y):
        self.eval()
        logits = self(X)
        pred = (logits > 0).long()
        return (pred == y).float().mean().item()