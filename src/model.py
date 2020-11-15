import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class Attention(pl.LightningModule):
    def __init__(self, d_model, num_heads, p, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input
            
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads
        
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)
        
        self.W_h = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(2)
        
        Q = Q / np.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(2, 3))
        
        A = nn.Softmax(dim=-1)(scores)
        
        H = torch.matmul(A, V)
        
        return H, A
        
    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
    def group_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size()
        
        Q = self.split_heads(self.W_q(X_q), batch_size)
        K = self.split_heads(self.W_k(X_k), batch_size)
        V = self.split_heads(self.W_v(X_v), batch_size)
        
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        H_cat = self.group_heads(H_cat, batch_size)
        
        H = self._h(H_cat)
        
        return H, A
    
class Encoder(pl.LightningModule):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        self.mha = Attention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
    
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        cnn_output = self.cnn(out1)
        out2 = self.layernorm2(out1 + cnn_output)
        
        return out2

