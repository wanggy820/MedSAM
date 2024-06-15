import torch
from torch import nn


class U2NetEncoder(nn.Module):
    def __init__(self, net, prompt_embed_dim, image_embedding_size, img_size=10124):
        super(U2NetEncoder, self).__init__()
        self.net = net
        self.img_size = img_size
        self.prompt_embed_dim = prompt_embed_dim
        self.image_embedding_size = image_embedding_size

    def forward(self, x):
        d0, d1, d2, d3, d4, d5, d6 = self.net(x)  # (3, 256, 64, 64)
        return d0.reshape(-1, self.prompt_embed_dim, self.image_embedding_size, self.image_embedding_size)
