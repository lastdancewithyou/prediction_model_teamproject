import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_Inv

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.use_norm = args.use_norm
        self.dropout = nn.Dropout(p = args.dropout)
        
        self.embedding = DataEmbedding_Inv(seq_len = args.seq_len,
                                           d_model = args.d_model,
                                           dropout = args.dropout)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = args.d_model,
                                       nhead = args.nhead
            ),
            num_layers = args.n_layers
        )
        
        self.fc1 = nn.Linear(args.d_model, args.d_model)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(args.d_model, 1, bias = True)
        
    def forward(self, x):
        
        # x: B x K x L
        
        if self.use_norm:
            means = x.mean(-1, keepdim = True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim = -1, keepdim = True, unbiased = False) + 1e-5)
            x /= stdev
        
        x = self.embedding(x) # B x K x d_model
        x = self.encoder(x) # B x K x d_model
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projector(x) # B x K x 1

        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x.permute(0, 2, 1)
            x = x * (stdev[:, :, 0].unsqueeze(1))
            x = x + (means[:, :, 0].unsqueeze(1))
            x = x.permute(0, 2, 1)
            
        return x # B x K x 1
        