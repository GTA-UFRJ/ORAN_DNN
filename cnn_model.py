import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import rn_model

class BrainConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, div=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.div = div
        if self.div <= 0:
            self.div = self.kernel_size-1
        self.pad = (self.div)//2
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv = nn.Conv1d(self.ch_in, self.ch_out, kernel_size=self.kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=self.ch_out)

    def forward(self, x):
        y = F.max_pool1d(torch.relu(self.batch_norm(self.conv(x))), self.div)
        return y

    def output_n(self, input_n):
        n = k_conv_out_n(1, input_n, self.kernel_size, self.div, self.pad)
        return (n, self.ch_out)


class ConvModel(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
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
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())
        self.conv_layers.append(nn.MaxPool1d(5))
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())        
        self.conv_layers.append(nn.MaxPool1d(2))
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())        
        self.conv_layers.append(nn.MaxPool1d(5))
        self.conv_layers.append(nn.Conv1d(7, 7, kernel_size=1))
        self.conv_layers.append(nn.ReLU())        
        self.conv_layers.append(nn.MaxPool1d(2))

        self.line_layers.append(nn.Linear(70, 18))
        self.line_layers.append(nn.Tanh())        
        self.line_layers.append(nn.Linear(18, 16))
        self.line_layers.append(nn.Tanh())        
        self.line_layers.append(nn.Linear(16, 3))
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            #print(x.shape)

        x = x.view(-1, 70)
        #print(x.shape)

        for linear_layer in self.line_layers:
            x = linear_layer(x)
            #print(x.shape)
        return x


    def extractInfTime(self, x):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        #This lines starts a timer to measure processing time
        starter.record()

        for layer in self.conv_layers:
            x = layer(x)
            #print(x.shape)

        x = x.view(-1, 70)
        #print(x.shape)

        for linear_layer in self.line_layers:
            x = linear_layer(x)
            #print(x.shape)

        ender.record()
        torch.cuda.synchronize()
        inf_time = starter.elapsed_time(ender)

        return inf_time


        #return self.model(x)
#x = torch.rand(1, 2, 20000)
#model = ConvModel()(x)
#rn_model.CharmBrain(chunk_size=20000)(x)