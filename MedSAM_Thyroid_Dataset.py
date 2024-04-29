
import numpy as np
import pandas as pd
import os
import cv2
import csv
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import json

sam = sam_model_registry['vit_b'](checkpoint='./work_dir/SAM/sam_vit_b_01ec64.pth')

class MedSAM_Thyroid_Dataset(Dataset):
    def __init__(self, data_dir, data_type="train"):
        self.data_dir = data_dir + "Thyroid_Dataset/tg3k/"

        with open(self.data_dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            if data_type == "train":
                self.names = data["train"]
            if data_type == "test":
                self.names = data["val"]

        self.transform = ResizeLongestSide(1024)
        self.preprocess = sam.preprocess
        self.img_size = sam.image_encoder.img_size
        self.resize = transforms.Resize((256, 256))

    def __len__(self):

        return len(self.names)

    def __getitem__(self, idx):

        #####################################
        data = self.names

        format = ".jpg"
        name = data[idx]
        image_path = self.data_dir + "Thyroid-image/" + "{:04d}".format(name) + format # 读取image data路径
        mask_path = self.data_dir + "Thyroid-mask/" + "{:04d}".format(name) + format # 读取mask data 路径
        #####################################

        img = cv2.imread(image_path) # 读取原图数据
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform.apply_image(img) #
        img = torch.as_tensor(img) # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0) # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img) # img nomalize or padding
        #####################################

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        mask = self.transform.apply_image(mask) # 变换(1024)

        mask = torch.as_tensor(mask) # torch tensor
        mask = mask.unsqueeze(0)

        h, w = mask.shape[-2:]

        padh = self.img_size - h
        padw = self.img_size - w

        mask = F.pad(mask, (0, padw, 0, padh))
        mask = self.resize(mask).squeeze(0)
        mask = (mask != 0) * 1

        #####################################

        data = {
            'image': img,
            'mask': mask,
        }

        return data

