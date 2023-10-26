import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_n(inputs, kernel_size, padding, stride, dilation=1):
    '''
    Compute the output size of a convolution or pool torch function.
    For convolution, stride should default to 1, for pooling, to kernel_size
    '''
    outs = inputs + 2*padding - dilation*(kernel_size-1) - 1
    outs = outs/stride + 1
    outs = math.floor(outs)
    return int(outs)


def k_conv_out_n(k, inputs, kernel_size, pool_kernel_size, padding):
    '''
    Compute the output size of a series of convolution (and optionally pooling) layers).
    '''
    n = 0
    out = inputs
    for i in range(k):
        out = conv_out_n(out, kernel_size, padding, 1)
        if pool_kernel_size > 0:
            out = conv_out_n(out, pool_kernel_size, 0, pool_kernel_size)
    return out


def params_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        self.conv = nn.Conv1d(self.ch_in, self.ch_out, kernel_size=self.kernel_size, padding=self.pad)
        self.batch_norm = nn.BatchNorm1d(num_features=self.ch_out)

    def forward(self, x):
        y = F.max_pool1d(torch.relu(self.batch_norm(self.conv(x))), self.div)
        return y

    def output_n(self, input_n):
        n = k_conv_out_n(1, input_n, self.kernel_size, self.div, self.pad)
        return (n, self.ch_out)


class BrainConvSkip(BrainConv):
    def forward(self, x):
        y = F.max_pool1d(torch.relu(self.batch_norm(self.conv(x))) + x, self.div)
        return y


class BrainLine(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.line = nn.Linear(inputs, outputs)

    def forward(self, x):
        y = torch.tanh(self.line(x))
        return y


class ResidualUnit(nn.Module):
    def __init__(self, chs, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (self.kernel_size-1)//2
        self.chs = chs
        self.conv1 = nn.Conv1d(self.chs, self.chs, kernel_size=self.kernel_size, padding=self.pad)
        self.conv2 = nn.Conv1d(self.chs, self.chs, kernel_size=self.kernel_size, padding=self.pad)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        z = x
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x) + z)
        return x

    def output_n(self, input_n):
        n = conv_out_n(input_n, self.kernel_size, self.pad, 1)
        n = conv_out_n(n, self.kernel_size, self.pad, 1)
        return (n, self.chs)


class ResidualStack(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, div=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.div = div
        if self.div <= 0:
            self.div = self.kernel_size-1
        self.pad = (self.div)//2
        self.ch_in = ch_in
        self.ch_out = ch_out

        self.conv = nn.Conv1d(self.ch_in, self.ch_out, kernel_size=1)
        self.res1 = ResidualUnit(self.ch_out, self.kernel_size)
        self.res2 = ResidualUnit(self.ch_out, self.kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = F.max_pool1d(torch.relu(x), self.div)
        return x

    def output_n(self, n):
        n = self.res1.output_n(n)[0]
        n = self.res2.output_n(n)[0]
        n = conv_out_n(n, self.div, 0, self.div)
        return (n, self.ch_out)


class CharmBrain(nn.Module):
    def __init__(self, chunk_size=20000):
        super().__init__()
        chs = 4  # convolution output channels
        self.conv_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()

        self.conv_layers.append(ResidualStack(2, chs, 3, 2))
        for _ in range(3):
            self.conv_layers.append(ResidualStack(chs, chs, 5, 5))
            self.conv_layers.append(ResidualStack(chs, chs, 3, 2))

        self.ll1_n = chunk_size
        for c in self.conv_layers:
            self.ll1_n = c.output_n(self.ll1_n)[0]
        self.ll1_n *= chs
        self.ll2_n = 16
        self.ll3_n = 16

        self.line_layers.append(BrainLine(self.ll1_n, self.ll2_n))
        self.line_layers.append(BrainLine(self.ll2_n, self.ll3_n))
        self.line_layers.append(BrainLine(self.ll3_n, 3))

        #print(f"Inner nodes: {self.ll1_n}")
        #print(f"Parameters: {params_count(self)}")

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            print(x.shape)

        x = x.view(-1, self.ll1_n)
        for layer in self.line_layers:
            x = layer(x)

        return x