import os
import cv2
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from U2_Net.data_loader import RescaleT
from U2_Net.data_loader import ToTensor
from U2_Net.data_loader import ToTensorLab
from U2_Net.data_loader import SalObjDataset

from U2_Net.model import U2NET  # full size version 173.6 MB
from U2_Net.model import U2NETP  # small version u2net 4.7 MB


if __name__ == "__main__":
    tensors = [torch.randn(3, 4) for _ in range(5)]  # 示例张量列表
    stacked = torch.stack(tensors, dim=0)  # 在第一个维度上合并
    print(stacked)