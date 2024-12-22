import math
from typing import List

import torch
from torch import nn
from functools import partial
from segment_anything.modeling import ImageEncoderViT
from torch.nn import functional as F

class PixelEncoder(nn.Module):
    def __init__(self,
                 img_size: int = 256,  # 图像大小
                 patch_size: int = 16,
                 embed_dim: int = 1024,  # patch 向量维度
                 depths: List[int] = [9, 7, 5, 3],  # VIT 深度
                 num_heads: int = 4,  # VIT 注意力深度
                 out_chans: int = 256,
                 ):
        super().__init__()

        self.pixel_layers = nn.ModuleList()
        self.image_size = img_size
        self.depths = depths
        for i in range(len(depths)):
            pixel = PixelEncoderLayer(img_size // (2**i), patch_size, embed_dim, depths[i], num_heads, out_chans)
            self.pixel_layers.insert(0, pixel)

    def forward(self, x):
        mask = []
        length = len(self.depths)
        x1 = F.interpolate(x, self.image_size // (2**(length - 1)), mode="bilinear", align_corners=False)

        for i in range(length):
            x1 = self.pixel_layers[i](x1)
            n, c , w, h = x1.shape
            w1 = int(math.sqrt(c*w*h))
            x1 = x1.reshape(n, 1, w1, w1)
            mask.append(x1)
            if i == length - 1:
                break
            x1 = F.interpolate(x1, w1 * 2, mode="bilinear", align_corners=False)

        return mask


class PixelEncoderLayer(nn.Module):
    def __init__(self,
                 img_size: int = 256,  # 图像大小
                 patch_size: int = 16,
                 embed_dim: int = 768,  # patch 向量维度
                 depth: int = 6,  # VIT 深度
                 num_heads: int = 8,  # VIT 注意力深度
                 out_chans: int = 256,  # 输出通道
                 ):
        super().__init__()
        self.in_chans = 3
        self.image_encoder = ImageEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            out_chans=out_chans,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            use_rel_pos=True,
            window_size=0,
        )

    def forward(self, x):
        x = x.repeat(1, self.in_chans, 1, 1)
        return self.image_encoder(x)
