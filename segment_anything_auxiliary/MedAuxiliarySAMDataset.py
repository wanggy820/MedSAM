import random
import cv2
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.nn import functional as F
from segment_anything.utils.transforms import ResizeLongestSide

class MedAuxiliarySAMDataset(Dataset):
    def __init__(self, sam, image_list, mask_list, bbox_shift=20):
        self.device = sam.device
        self.preprocess = sam.preprocess

        self.image_list = image_list
        self.mask_list = mask_list

        self.originSize = 1200
        self.transform_1200 = ResizeLongestSide(self.originSize)

        self.image_size = sam.image_encoder.img_size
        self.transform_1024 = ResizeLongestSide(self.image_size)

        self.output_size = 256
        self.transform_256 = ResizeLongestSide(self.output_size)

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
        image = io.imread(image_path)  # 读取原图数据
        mask_np = io.imread(mask_path)  # 读取掩码数据
        if 3==len(mask_np.shape):
            mask_np = mask_np[:,:0]

        if (3 == len(image.shape) and 2 == len(mask_np.shape)):
            mask_np = mask_np[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(mask_np.shape)):
            image = image[:, :, np.newaxis]
            mask_np = mask_np[:, :, np.newaxis]
        #####################################

        image_1200 = self.transform_1200.apply_image(image)  #
        h, w = image_1200.shape[:2]
        if h > w:
            new_h = self.image_size
            new_w = int(new_h*w/h)
        else:
            new_w = self.image_size
            new_h = int(new_w*h/w)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_1024 = image_1200[top: top + new_h, left: left + new_w]
        image_1024 = torch.as_tensor(image_1024)  # torch tensor 变更
        image_1024 = image_1024.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None
        image_1024 = self.preprocess(image_1024.to(device=self.device))  # img nomalize or padding

        train_image = image_1024.unsqueeze(1).float()
        train_image = torch.nn.functional.interpolate(train_image, scale_factor=self.output_size/self.image_size, mode='bilinear',
                                                     align_corners=False)
        image_256 = train_image.squeeze(1)


        #####################################

        mask_1200 = self.transform_1200.apply_image(mask_np)  #
        mask_1024 = mask_1200[top: top + new_h, left: left + new_w]
        mask_1024 = self.preprocessMask(mask_1024, self.transform_1024, self.image_size)
        mask_256 = self.transform_256.apply_image(mask_1024)
        mask_256 = torch.as_tensor(mask_256).unsqueeze(0)
        #####################################
        if random.random() >= 0.5:
            image_1024 = image_1024[::-1]
            mask_256 = mask_1024[::-1]

        h, w = mask_np.shape[-2:]
        size = np.array([w, h])
        data = {
            'image_1024': image_1024,
            'image_256': image_256,
            'mask': mask_256,
            "image_path": image_path,
            "mask_path": mask_path,
            "size": size
        }
        return data