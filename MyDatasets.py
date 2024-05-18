import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from skimage import transform, io


class MyDatasets(Dataset):
    def __init__(self, image_list, mask_list, prompt_bboxes=None, prompt_masks=None, data_type="train", bbox_shift=20):
        self.image_list = image_list
        self.mask_list = mask_list
        self.prompt_bboxes = prompt_bboxes
        self.prompt_masks = prompt_masks
        self.data_type = data_type
        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        ######################################
        image_path = self.image_list[idx]  # 读取image data路径
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        img_1024 = transform.resize(
            img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1)
        )
        ######################################

        if len(self.mask_list) == 0:
            mask_3c = None
            mask_tensor = None
        else:
            mask_path = self.mask_list[idx]  # 读取mask data 路径
            mask_3c = io.imread(mask_path)
            mask_tensor = torch.tensor(mask_3c).float()
        ######################################

        box_np = None
        mask_np = None
        H, W, _ = img_3c.shape
        if self.data_type == "train":
            y_indices, x_indices = np.where(mask_3c > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            box_np = np.array([x_min, y_min, x_max, y_max])
            mask_np = mask_3c
        else:
            if self.prompt_bboxes is not None and len(self.prompt_bboxes) > 0:
                box_np = self.prompt_bboxes[idx]

            if self.prompt_masks is not None and len(self.prompt_masks) > 0:
                mask_np = self.prompt_masks[idx]

        prompt_box = None
        if box_np is not None:
            box_1024 = box_np / np.array([W, H, W, H]) * 1024
            prompt_box = torch.tensor(box_1024).float()
        ######################################
        prompt_masks = None
        if mask_np is not None:
            mask_256 = transform.resize(
                mask_3c, (256, 256), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            mask_256 = (mask_256 - mask_256.min()) / np.clip(
                mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 1)
            # prompt_masks = np.expand_dims(mask_256, axis=0)
            prompt_masks = np.expand_dims(mask_256, axis=0).astype(np.float32)

        #####################################
        data = {
            'image': img_1024_tensor,
            "image_path": image_path,
            "height": H,
            "width": W
        }

        if mask_3c is not None:
            data["mask"] = mask_tensor
        if prompt_box is not None:
            data["prompt_box"] = prompt_box
        if mask_np is not None:
            data["prompt_masks"] = prompt_masks

        return data
