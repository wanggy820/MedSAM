import torch
from torch import nn


class U2NetEncoder(nn.Module):
    def __init__(self, net, img_size=10124):
        super(U2NetEncoder, self).__init__()
        self.net = net
        self.img_size = img_size

    def forward(self, x):
        d0, d1, d2, d3, d4, d5, d6 = self.net(x)  # (3, 256, 64, 64)
        return d0.reshape(256, 64, 64)
