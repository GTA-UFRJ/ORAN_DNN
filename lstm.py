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



class LSTM(nn.Module):
    def __init__(self, input_size, num_layers):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size

        #self.hidden_size = hidden_size

        self.lstm_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()
        
        self.lstm_layers.append(LSTMStack(2, 4, num_layers, div=2))

        for _ in range(3):
            self.lstm_layers.append(LSTMStack(4, 4, num_layers, div=5))
            self.lstm_layers.append(LSTMStack(4, 4, num_layers, div=2))

        self.line_layers.append(BrainLine(40, 16))
        self.line_layers.append(BrainLine(16, 16))
        self.line_layers.append(BrainLine(16, 3))

        #nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #num_layers=num_layers, batch_first=True)        

        #self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size,
        #num_layers=num_layers, batch_first=True)) #lstm
       

    def forward(self, x):

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.input_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.input_size)) #internal state

        for lstm_layer in self.lstm_layers:

            x, h_0, c_0 = lstm_layer(x, h_0, c_0) #lstm with input, hidden, and internal state

        x = x.view(-1, 40)



        for layer in self.line_layers:
            x = layer(x)

        return x

#x = torch.rand(1, 2, 20000)
#model = LSTM(input_size=2, num_layers=2)(x)


