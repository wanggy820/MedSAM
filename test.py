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
from PIL import Image, ImageFilter
import glob

from U2_Net.data_loader import RescaleT
from U2_Net.data_loader import ToTensor
from U2_Net.data_loader import ToTensorLab
from U2_Net.data_loader import SalObjDataset

from U2_Net.model import U2NET  # full size version 173.6 MB
from U2_Net.model import U2NETP  # small version u2net 4.7 MB

image = cv2.imread("./save_models/MICCAI/vit_b_3_1.02/val/A-90.png")
smoothed = cv2.blur(image, (5, 5))
a = torch.from_numpy(smoothed > 127)*1
b = torch.where(a >= 1, 255.0, 0)
smoothed1 = b.numpy()
cv2.imshow("origin", image)
cv2.imshow("smoothed", smoothed1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image = Image.open("./save_models/MICCAI/vit_b_3_1.02/val/A-90.png")
# en = image.filter(ImageFilter.SMOOTH_MORE)
# # image.show()
# en.show()




if __name__ == "__main__":
    tensors = [torch.randn(3, 4) for _ in range(5)]  # 示例张量列表
    stacked = torch.stack(tensors, dim=0)  # 在第一个维度上合并
    print(stacked)