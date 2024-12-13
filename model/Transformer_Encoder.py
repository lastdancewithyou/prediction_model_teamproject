import torch
import torch.nn as nn

from layers.Embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.c_in = args.c_in
        self.use_norm = args.use_norm
        self.dropout = nn.Dropout(p = args.dropout)
        
        self.embedding = DataEmbedding(
            c_in = args.c_in,
            d_model = args.d_model
        )

        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = args.d_model,
                                       nhead = args.nhead,
                                       batch_first = True
            ),
            num_layers = args.n_layers
        )
        
        self.norm = nn.LayerNorm(args.d_model)
        self.fc1 = nn.Linear(args.d_model, args.d_model)
        self.relu = nn.ReLU()
        self.projector = nn.LazyLinear(1)
        
    def forward(self, x, padding_mask = None):
        
        # x: B x K x L
        
        if self.use_norm:
            x = x.permute(0, 2, 1)
            means = x.mean(-1, keepdim = True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim = -1, keepdim = True, unbiased = False) + 1e-5)
            x /= stdev
            x = x.permute(0, 2, 1)

        x = self.embedding(x).permute(0, 2, 1) # B x L x d_model
        
        if padding_mask is not None:
            x = self.encoder(x, src_key_padding_mask=padding_mask)  # B x L x d_model
        else:
            x = self.encoder(x)  # B x L x d_model
            
        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projector(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.reshape(x.shape[0], 1, -1)
        

        
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            x = x.permute(0, 2, 1)
            x = x * (stdev[:, :, 0].unsqueeze(1))
            x = x + (means[:, :, 0].unsqueeze(1))
            x = x.permute(0, 2, 1)
            
        return x
        