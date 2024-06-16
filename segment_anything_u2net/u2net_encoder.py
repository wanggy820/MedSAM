import torch.nn.functional as F
from torch import nn


class U2NetEncoder(nn.Module):
    def __init__(self, net, image_embedding_size, img_size=10124):
        super(U2NetEncoder, self).__init__()
        self.net = net
        self.img_size = img_size
        self.image_embedding_size = image_embedding_size

    def forward(self, x):
        y = F.interpolate(x, size=(self.image_embedding_size, self.image_embedding_size), mode="bilinear", align_corners=False,)
        d0, d1, d2, d3, d4, d5, d6 = self.net(y)  # (-1, 256, 64, 64)
        return d0
