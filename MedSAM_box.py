import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAMBox(Dataset):
    def __init__(self, sam, image_list, mask_list, auxiliary_list, bbox_shift=0):
        self.device = sam.device
        self.preprocess = sam.preprocess

        self.image_list = image_list
        self.mask_list = mask_list
        self.auxiliary_list = auxiliary_list

        self.img_size = sam.image_encoder.img_size
        self.transform_image = ResizeLongestSide(self.img_size)

        self.output_size = 256
        self.transform_mask = ResizeLongestSide(self.output_size)

        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.image_list)

    def preprocessMask(self, mask_np, transform, size):
        mask = transform.apply_image(mask_np)  #
        mask = torch.as_tensor(mask/255.0)
        h, w = mask.shape[-2:]
        padh = size - h
        padw = size - w
        mask = F.pad(mask, (0, padw, 0, padh))
        return mask


    def __getitem__(self, idx):
        image_path = self.image_list[idx]  # 读取image data路径
        mask_path = self.mask_list[idx]  # 读取mask data 路径
        auxiliary_path = self.auxiliary_list[idx]
        #####################################

        img = cv2.imread(image_path)  # 读取原图数据
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform_image.apply_image(img)  #
        img = torch.as_tensor(img)  # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img.to(device=self.device))  # img nomalize or padding
        #####################################

        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        mask_256 = self.preprocessMask(mask_np, self.transform_mask, self.output_size)
        mask_256 = torch.as_tensor(mask_256).unsqueeze(0)

        ##################################### 不能用 find_bboxes() 张量维度不一样
        auxiliary_np = cv2.imread(auxiliary_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        auxiliary_256 = self.preprocessMask(auxiliary_np, self.transform_mask, self.output_size)
        auxiliary_1024 = self.preprocessMask(auxiliary_np, self.transform_image, self.img_size)

        y_indices, x_indices = np.where(auxiliary_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            x_min = y_min = 0
            x_max = y_max = self.img_size
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(self.img_size, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(self.img_size, y_max + random.randint(0, self.bbox_shift))

        box_1024 = np.array([x_min, y_min, x_max, y_max])
        box_1024 = box_1024.astype(np.int16)

        #####################################
        bbox_shift = random.randint(0, self.bbox_shift * 2)
        ratio = min(bbox_shift / (x_max - x_min) + 1, 1.5)
        ratio = max(ratio, 1)
        size = int(self.output_size * ratio)
        transform_ratio = ResizeLongestSide(size)
        auxiliary_ratio = self.preprocessMask(auxiliary_np, transform_ratio, size)

        top = int(size * (ratio - 1) / 2)
        bottom = top + self.output_size
        left = int(size * (ratio - 1) / 2)
        right = left + self.output_size
        auxiliary_ratio_masks = auxiliary_ratio[top:bottom, left:right]

        auxiliary_ratio_masks += auxiliary_256
        auxiliary_ratio_masks = auxiliary_ratio_masks // 2 + auxiliary_ratio_masks % 2
        prompt_masks = torch.where(auxiliary_ratio_masks > 0, 1.0, 0.0).unsqueeze(0)
        #####################################
        h, w = mask_np.shape[-2:]
        size = np.array([w, h])
        data = {
            'image': img,
            'mask': mask_256,
            "prompt_box": box_1024,
            "prompt_masks": prompt_masks,
            "image_path": image_path,
            "mask_path": mask_path,
            "size": size
        }
        return data