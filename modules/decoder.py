import torch
from torch import nn

from vit import ViT


class Decoder(nn.Module):
    def __init__(self, d_model, d_patch, codebook_size, seqlen, n_heads, n_blocks, dff, dropout):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model))
        self.vit = ViT(d_model, seqlen, n_heads, n_blocks, dff, dropout)
        self.outproj = nn.Linear(d_model, d_patch)
        
    def forward(self, ze):
        ze = ze.unsqueeze(-2)
        idx = (ze - self.codebook).norm(2, dim=-1).argmin(-1)
        zq = self.codebook[idx]
        x = self.vit(zq)
        logits = self.outproj(x)
        return logits
    