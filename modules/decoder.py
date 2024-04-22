import torch
from torch import nn

from modules.vit import ViT


class Decoder(nn.Module):
    def __init__(self, d_model, d_patch, codebook_size, seqlen, n_heads, n_blocks, dff, dropout):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model))
        self.vit = ViT(d_model, seqlen, n_heads, n_blocks, dff, dropout)
        self.outproj = nn.Linear(d_model, d_patch)
        
    def forward(self, ze):
        _ze = ze.detach().unsqueeze(-2)
        idx = (_ze - self.codebook).norm(2, dim=-1).argmin(-1)
        zq = self.codebook[idx]
        if self.training:
            x = (zq - ze).detach() + ze
        else:
            x = zq
        x = self.vit(x)
        logits = self.outproj(x)
        return logits, zq
            