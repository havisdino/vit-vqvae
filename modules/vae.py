from torch import nn

from modules.decoder import Decoder
from modules.encoder import Encoder


class VAE(nn.Module):
    def __init__(self, d_model, d_patch, dictsize, seqlen, n_heads, n_blocks, dff, dropout):
        super().__init__()
        self.encoder = Encoder(d_model, d_patch, seqlen, n_heads, n_blocks, dff, dropout)
        self.decoder = Decoder(d_model, d_patch, dictsize, seqlen, n_heads, n_blocks, dff, dropout)
        
    def forward(self, x):
        ze = self.encoder(x)
        