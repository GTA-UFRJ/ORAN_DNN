import math
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import rn_model
from torch.autograd import Variable


class BrainLine(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.line = nn.Linear(inputs, outputs)

    def forward(self, x):
        y = torch.tanh(self.line(x))
        return y


class LSTMStack(nn.Module):
    def __init__(self, ch_in, ch_out, num_layers, div=0):
        super().__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.div = div
        self.num_layers = num_layers

        self.conv = nn.Conv1d(self.ch_in, self.ch_out, kernel_size=1)
        self.lstm = nn.LSTM(input_size=self.ch_out, hidden_size=self.ch_out, 
            num_layers=num_layers, batch_first=True)

        #self.res1 = ResidualUnit(self.ch_out, self.kernel_size)
        #self.res2 = ResidualUnit(self.ch_out, self.kernel_size)

    def forward(self, x, h, c):

        x = self.conv(x)
        x = x.permute(0, 2, 1)
        
        x, (h, c) = self.lstm(x)
        x = x.permute(0, 2, 1)
        #x = self.res2(x)
        x = F.max_pool1d(torch.relu(x), self.div)
        return x, h, c

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()

        self.conv_layers.append(nn.Conv1d(2, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool1d(2))
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool1d(5))
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool1d(2))

        self.lstm_layers.append(LSTMStack(7, 7, 2, div=5))
        self.lstm_layers.append(LSTMStack(7, 7, 2, div=2))

        self.lstm_layers.append(LSTMStack(7, 7, 2, div=5))
        self.lstm_layers.append(LSTMStack(7, 7, 2, div=2))

        self.line_layers.append(rn_model.BrainLine(70, 16))
        self.line_layers.append(rn_model.BrainLine(16, 16))
        self.line_layers.append(rn_model.BrainLine(16, 3))

    def forward(self, x):

        h = Variable(torch.zeros(2, x.size(0), 2)) #hidden state
        c = Variable(torch.zeros(2, x.size(0), 2)) #internal state

        for layer in self.conv_layers:
            x = layer(x)
            print(x.shape)

        for layer in self.lstm_layers:
            x, h, c = layer(x, h, c)
            print(x.shape)

        for layer in self.line_layers:
            x = layer(x)

x = torch.rand(1, 2, 20000)
model = ConvLSTM()(x)