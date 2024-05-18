import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from skimage import transform, io


class MyDatasets(Dataset):
    def __init__(self, sam, image_list, mask_list, bboxes=None, masks=None, data_type="train", bbox_shift=20):
        self.sam = sam
        self.image_list = image_list
        self.mask_list = mask_list
        self.bboxes = bboxes
        self.data_type = data_type
        self.masks = masks
        self.bbox_shift = bbox_shift

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
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
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.sam.device)
        )

        embedding = self.sam.image_encoder(
            img_1024_tensor
        )  # (1, 256, 64, 64)

        H, W, _ = img_3c.shape
        if len(self.mask_list) == 0:
            mask_3c = None
        else:
            mask_path = self.mask_list[idx]  # 读取mask data 路径
            mask_3c = io.imread(mask_path)

        box_np = None
        mask_np = None
        if self.data_type == "train":
            y_indices, x_indices = np.where(mask_3c > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            box_np = np.array([[x_min, y_min, x_max, y_max]])
            mask_np = mask_3c
        else:
            if self.bboxes is not None and len(self.bboxes) > 0:
                box_np = self.bboxes[idx]

            if self.masks is not None and len(self.masks) > 0:
                mask_np = self.masks[idx]

        box_1024 = None
        if box_np is not None:
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

        mask_256 = None
        if mask_np is not None:
            mask_256 = transform.resize(
                mask_3c, (256, 256), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            mask_256 = (mask_256 - mask_256.min()) / np.clip(
                mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 1)
            mask_256 = np.expand_dims(mask_256, axis=0)
            mask_256 = np.expand_dims(mask_256, axis=0)

        sam_mask = self.medsam_inference(embedding, box_1024, mask_256, H, W)
        normalized_data = (sam_mask * 255 / np.max(sam_mask)).astype(np.uint8)
        # cv2.imwrite('tt.png', normalized_data)
        pre_mask = torch.tensor(normalized_data).float()

        mask_3c = torch.tensor(mask_3c).float()
        #####################################
        data = {
            'image': pre_mask,
            'mask': mask_3c,
            "image_path": image_path,
        }
        return data

    def medsam_inference(self, img_embed, box_1024, mask_1024, height, width):
        box_torch = None
        if box_1024 is not None:
            box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

        mask_torch = None
        if mask_1024 is not None:
            mask_torch = torch.as_tensor(mask_1024, dtype=torch.float, device=img_embed.device)
            if len(box_torch.shape) == 2:
                mask_torch = box_torch[:, None, :]  # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=mask_torch,
        )
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

        low_res_pred = F.interpolate(
            low_res_pred,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )  # (1, 1, gt.shape)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        return medsam_seg
