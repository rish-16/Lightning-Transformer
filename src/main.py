import torch 
from torch import nn
import numpy as np

from model import Attention, Encoder

mha = Attention(d_model=512, num_heads=8, p=0)
encoder = Encoder(d_model=512, num_heads=8, conv_hidden_dim=128)

def print_out(Q, K, V):
    temp_out, temp_attn = mha.scaled_dot_product_attention(Q, K, V)
    print('Attention weights are:', temp_attn.squeeze())
    print('Output is:', temp_out.squeeze())
    
test_K = torch.tensor(
    [[10, 0, 0],
     [ 0,10, 0],
     [ 0, 0,10],
     [ 0, 0,10]]
).float()[None,None]

test_V = torch.tensor(
    [[   1,0,0],
     [  10,0,0],
     [ 100,5,0],
     [1000,6,0]]
).float()[None,None]

test_Q = torch.tensor(
    [[0, 0, 10], [0, 10, 0], [10, 10, 0]]
).float()[None,None]

print_out(test_Q, test_K, test_V)