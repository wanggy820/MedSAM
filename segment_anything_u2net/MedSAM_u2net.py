import random

from skimage import transform
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from segment_anything.utils.transforms import ResizeLongestSide

class MedSAM_U2net(Dataset):
    def __init__(self, sam, image_list, mask_list, bbox_shift=0, ratio=1.01):
        self.device = sam.device
        self.image_list = image_list
        self.mask_list = mask_list

        self.transform = ResizeLongestSide(1024)
        self.preprocess = sam.preprocess
        self.img_size = sam.image_encoder.img_size
        self.resize = transforms.Resize((256, 256))
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

        img = self.transform.apply_image(img)  #
        img = torch.as_tensor(img)  # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img.to(device=self.device))  # img nomalize or padding
        #####################################

        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        H, W = mask_np.shape
        h = int(H * self.ratio)
        w = int(W * self.ratio)
        mask_ratio = transform.resize(
            mask_np, (h, w), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        top = int(h*(self.ratio - 1)/2)
        bottom = top + H
        left = int(w * (self.ratio - 1) / 2)
        right = left + W
        mask_ratio_np = mask_ratio[top:bottom, left:right]
        mask_ratio_np += mask_np
        mask_ratio_np = mask_ratio_np // 2 + mask_ratio_np % 2
        mask_ratio_masks = torch.as_tensor(mask_ratio_np, dtype=torch.float32)  # torch tensor
        mask_ratio_masks = mask_ratio_masks.unsqueeze(0)

        mask = torch.as_tensor(mask_np/255)  # torch tensor
        mask = mask.unsqueeze(0)

        ##################################### 不能用 find_bboxes() 张量维度不一样
        y_indices, x_indices = np.where(mask_np > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        box_np = np.array([x_min, y_min, x_max, y_max])
        H, W = mask_np.shape[-2:]
        box_1024 = box_np / np.array([W, H, W, H]) * 1024
        box_1024 = box_1024.astype(np.int16)
        #####################################
        mask_256 = transform.resize(
            mask_np, (256, 256), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        mask_256 = (mask_256 - mask_256.min()) / np.clip(
            mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 1)
        prompt_masks = np.expand_dims(mask_256, axis=0).astype(np.float32)

        data = {
            'image': img,
            'mask': mask,
            "prompt_box": box_1024,
            "prompt_masks": prompt_masks,
            "image_path": image_path,
            "mask_path": mask_path,
            "mask_ratio_masks": mask_ratio_masks
        }
        return data