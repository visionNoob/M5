import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=3062, hidden_layer_size=100, output_size=28):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        lstm_out = lstm_out[:, -1]
        predictions = self.linear(lstm_out)

        return predictions