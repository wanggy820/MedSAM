import random

import torchvision
from skimage import transform
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAMBox(Dataset):
    def __init__(self, sam, image_list, mask_list, bbox_shift=0, ratio=1.05):
        self.device = sam.device
        self.preprocess = sam.preprocess

        self.image_list = image_list
        self.mask_list = mask_list
        self.img_size = sam.image_encoder.img_size
        self.output_size = 256
        self.bbox_shift = bbox_shift
        self.ratio = max(ratio, 1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]  # 读取image data路径
        mask_path = self.mask_list[idx]  # 读取mask data 路径
        #####################################

        img = cv2.imread(image_path)  # 读取原图数据
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = torch.as_tensor(img)  # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img.to(device=self.device))  # img nomalize or padding
        #####################################

        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        mask_256_np = cv2.resize(mask_np, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        mask_256 = torch.as_tensor(mask_256_np).unsqueeze(0)  # torch tensor

        ##################################### 不能用 find_bboxes() 张量维度不一样
        mask_1024 = transform.resize(
            mask_np, (self.img_size, self.img_size), order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        y_indices, x_indices = np.where(mask_1024 > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(self.img_size, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(self.img_size, y_max + random.randint(0, self.bbox_shift))
        box_1024 = np.array([x_min, y_min, x_max, y_max])
        box_1024 = box_1024.astype(np.int16)

        #####################################
        size = int(self.output_size * self.ratio)
        mask_ratio = transform.resize(
            mask_np, (size, size), order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        top = int(size * (self.ratio - 1) / 2)
        bottom = top + self.output_size
        left = int(size * (self.ratio - 1) / 2)
        right = left + self.output_size
        mask_ratio_np = mask_ratio[top:bottom, left:right]
        mask_ratio_np += mask_256_np
        mask_ratio_np = mask_ratio_np // 2 + mask_ratio_np % 2
        mask_ratio_masks = torch.as_tensor(mask_ratio_np, dtype=torch.float32)  # torch tensor
        prompt_masks = torch.where(mask_ratio_masks > 0, 1.0, 0.0).unsqueeze(0)
        #####################################

        data = {
            'image': img,
            'mask': mask_256,
            "prompt_box": box_1024,
            "prompt_masks": prompt_masks,
            "image_path": image_path,
            "mask_path": mask_path,
        }
        return data