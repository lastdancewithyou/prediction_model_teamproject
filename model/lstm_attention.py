import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        input_size = self.args.input_size
        hidden_size = self.args.hidden_size
        fc_hidden_size = self.args.fc_hidden_size
        dropout =self.args.dropout

        # LSTM Layer = hidden_size 256 짜리 LSTM layer 2개
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=dropout)

        # Soft Attention mechanism layer
        self.attention = nn.Linear(hidden_size, 1)  # Produces scalar attention score for each time step

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size)
        """
        (진혁) 논문에서는 fc-layer를 3개 썼다고 했는데, 모델 구조 설명하는 부분에서 attention layer 다음에 fc-layer를
        두개 사용했다고 나와있어서 일단 2개로 해놓았습니다. 추후 변경 가능
        """
        # Activation and dropout
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # For normalizing attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_lens):
        """
        x: (batch_size, variable_num, sequence_length)
        seq_lens: padding 제외 실제 시퀀스 길이 (배치별)
        """
        x = x.permute(0, 2, 1)  # LSTM input dimension에 맞게 차원조정 (batch_size, sequence_length, variable_num)

        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h_n1, c_n1) = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # shape: (batch_size, seq_lens, hidden_size)


        attention_scores = self.attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_len)
        attention_weights = self.softmax(attention_scores)  # softmax로 attention_weights 계산
        weighted_lstm_out = lstm_out * attention_weights.unsqueeze(-1)
        context_vector = torch.sum(weighted_lstm_out, dim=1)  # context_vector 계산


        fc_out = self.fc1(context_vector)
        fc_out = self.relu(fc_out)
        fc_out = self.dropout(fc_out)
        output = self.fc2(fc_out)


        return output