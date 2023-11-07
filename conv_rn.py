import math
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
import rn_model
from torch.autograd import Variable

class ConvRNN(nn.Module):
    def __init__(self):
        super(ConvRNN, self).__init__()

        self.conv_layers = nn.ModuleList()
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

        self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

        self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

        self.line_layers.append(rn_model.BrainLine(70, 16))
        self.line_layers.append(rn_model.BrainLine(16, 16))
        self.line_layers.append(rn_model.BrainLine(16, 3))
        
        #for _ in range(3):
        #    self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        #    self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, 70)

        for layer in self.line_layers:
            x = layer(x)
            
        return x




class ConvRNN2(nn.Module):
    def __init__(self):
        super(ConvRNN2, self).__init__()

        self.conv_layers = nn.ModuleList()
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


        self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

        self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

        self.line_layers.append(rn_model.BrainLine(70, 16))
        self.line_layers.append(rn_model.BrainLine(16, 16))
        self.line_layers.append(rn_model.BrainLine(16, 3))
        
        #for _ in range(3):
        #    self.conv_layers.append(rn_model.ResidualStack(7, 7, 5, 5))
        #    self.conv_layers.append(rn_model.ResidualStack(7, 7, 3, 2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(-1, 70)

        for layer in self.line_layers:
            x = layer(x)
            
        return x





x = torch.rand(1, 2, 20000)
model = ConvRNN2()(x)