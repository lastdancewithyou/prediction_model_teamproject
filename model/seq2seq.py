import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 2, batch_first=True)
    
    def forward(self, x, lengths):

        x = x.permute(0, 2, 1)  # LSTM input dimension에 맞게 차원조정 (batch_size, sequence_length, variable_num)

        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_x)

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, encoder_outputs, hidden, mask=None):

        
        hidden = hidden[-1].unsqueeze(1)  
        scores = torch.tanh(self.attention(encoder_outputs))  
        scores = scores @ self.v  

        if mask is not None:
            
            scores = scores.masked_fill(mask == 0, float('-inf'))

        
        attention_weights = F.softmax(scores, dim=1)  

        
        context_vector = torch.sum(attention_weights.unsqueeze(2) * encoder_outputs, dim=1)  
        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, context_vector):

        output, _ = self.lstm(context_vector.unsqueeze(1))
        output = self.fc(output.squeeze(1))
        return output


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        input_size = self.args.input_size
        hidden_size = self.args.hidden_size
        output_size = self.args.fc_hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(hidden_size, output_size)
    
    def forward(self, x, lengths):

        encoder_outputs, hidden, cell = self.encoder(x, lengths)
        

        batch_size, seq_len, _ = encoder_outputs.size()
        mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


        context_vector, _ = self.attention(encoder_outputs, hidden, mask)
        

        outputs = self.decoder(context_vector)
        return outputs