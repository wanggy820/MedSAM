import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class U2NetEncoder(nn.Module):
    def __init__(self, net, img_size):
        super(U2NetEncoder, self).__init__()
        self.net = net
        self.img_size = img_size

    def forward(self, x):  # 3 * 1024 * 1024
        # y : 3 * 64 * 64
        d0, d1, d2, d3, d4, d5, d6 = self.net(x)  # (-1, 256, 64, 64)
        # ret = torch.cat((d0, d1, d2, d3), dim=1)
        ret = d0.reshape(-1, 256, 64, 64)
        return ret
