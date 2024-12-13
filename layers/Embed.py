import torch
import torch.nn as nn
import math

class DataEmbedding_Inv(nn.Module):
    """_summary_
    
    Inverted Transformer Embedding
    
    seq_len을 embedding dimension으로 보냄
    """
    def __init__(self, seq_len, d_model, dropout = 0.1):
        super(DataEmbedding_Inv, self).__init__()
        
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x, mask = None):
        # x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)