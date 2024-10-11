import torch
import torch.nn as nn

class LSTM(nn.Module):
    #Main model
    def __init__(self, input_size=2, hidden_size=400, num_layers=3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM from pytorch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Linear layer to turn last hidden state into a single number
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # init hidden and cell tensors
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        # forward pass
        out, _ = self.lstm(x, (h0, c0))
        # running the last hidden state on the linear layer
        out = self.fc(out[:, -1, :])
        return out
