import numpy as np
import torch.utils.data as data
import pandas as pd
import torchvision
from PIL import Image
from torchvision import transforms


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class ISBIDataset(data.Dataset):
    cmap = voc_cmap()
    def __init__(self, filePath, direction=None, transform=None, needSqueeze=True):
        super(ISBIDataset, self).__init__()
        self.filePath = filePath
        if direction is not None:
            self.filePath = direction + self.filePath
        f = open(self.filePath, encoding="utf-8")
        self.data = pd.read_csv(f)
        self.transform = transform
        self.direction = direction
        self.needSqueeze = needSqueeze
    def __getitem__(self, item):
        img_path = self.data["img"][item]
        seg_path = self.data["seg"][item]

        img = Image.open(self.direction + "/datasets/" + img_path)
        seg = Image.open(self.direction + "/datasets/" + seg_path)

        # print("img_path:{}, {}, seg_path:{}, {}".format(img_path, img.size, seg_path, seg.size))
        img = self.transform(img)
        seg = self.transform(seg)
        if self.needSqueeze is True:
            seg = seg.squeeze()
        return img, seg

    def __len__(self):
        return len(self.data["img"])


    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

