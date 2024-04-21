import torch
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        assert d_model % nheads == 0
        
        self.nheads = nheads
        self.dim_head = d_model // nheads
        self.d_model = d_model
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        self.register_buffer('scale', torch.FloatTensor([self.dim_head]).sqrt())
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, attn_mask=None):
        B, L, _ = inputs.size()
        
        qkv = self.qkv(inputs)
        qkv = qkv.view(B, L, self.nheads, -1)
        qkv = qkv.permute(0, 2, 1, 3)
        Q, K, V = qkv.split(self.dim_head, dim=-1)
        scores = Q.matmul(K.permute(0, 1, 3, 2)) / self.scale
        if attn_mask is not None:
            scores += attn_mask
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        outputs = scores.matmul(V)
        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        outputs = outputs.view(B, L, -1)
        
        return outputs
    
    
class FFN(nn.Sequential):
    def __init__(self, d_model, dff):
        super().__init__()
        self.append(nn.Linear(d_model, dff))
        self.append(nn.GELU())
        self.append(nn.Linear(dff, d_model))
        
        
class ReZeroTransformerBlock(nn.Module):
    def __init__(self, d_model, nheads, dff, dropout):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.self_attn = SelfAttention(d_model, nheads, dropout)
        self.ffn = FFN(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, attn_mask=None):
        x = self.self_attn(inputs, attn_mask) * self.alpha + inputs
        x = self.dropout(x)
        x = self.ffn(x) * self.alpha + x
        x = self.dropout(x)
        return x
    
    
class ViT(nn.Module):
    def __init__(self, d_model, seqlen, n_heads, n_blocks, dff, dropout):
        super().__init__()

        self.pe = nn.Parameter(torch.randn(1, seqlen, d_model))
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(ReZeroTransformerBlock(d_model, n_heads, dff, dropout))
        
    def forward(self, x):
        x += self.pe
        for block in self.blocks:
            x = block(x)
        return x
