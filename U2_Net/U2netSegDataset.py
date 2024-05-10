# coding: utf-8
# author: hxy
# 2022-04-20
"""
数据读取dataset
"""
import cv2
import numpy as np
from torch.utils.data import Dataset


# dataset for u2net
class U2netSegDataset(Dataset):
    def __init__(self, image_list, mask_list, input_size=(320, 320)):
        """
        :param image_list: 数据集图片文件夹路径
        :param mask_list: 数据集mask文件夹路径
        :param input_size: 图片输入的尺寸
        """

        self.image_list = image_list
        self.mask_list = mask_list
        self.input_size = input_size
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.input_size)
        img2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2norm = (img2rgb - self.mean) / self.std
        # 图像格式改为nchw
        img2nchw = np.transpose(img2norm, [2, 0, 1]).astype(np.float32)

        mask_path = self.mask_list[index]
        gt_mask = cv2.imread(mask_path)
        gt_mask = cv2.resize(gt_mask, self.input_size)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        gt_mask = gt_mask / 255.
        gt_mask = np.expand_dims(gt_mask, axis=0)

        return img2nchw, gt_mask
