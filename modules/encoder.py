import torch
from torch import nn

from vit import ViT


class Encoder(nn.Module):
    def __init__(self, d_model, d_patch, seqlen, n_heads, n_blocks, dff, dropout):
        super().__init__()
        self.inproj = nn.Linear(d_patch, d_model)
        self.vit = ViT(d_model, seqlen, n_heads, n_blocks, dff, dropout)
        self.outproj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        x = self.inproj(x)
        x = self.vit(x)
        ze = self.outproj(x)
        return ze
    