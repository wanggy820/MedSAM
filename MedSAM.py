import glob
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import random
join = os.path.join

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


class ISBIDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root

        filePath = data_root + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
        f = open(filePath, encoding="utf-8")
        data = pd.read_csv(f)
        self.tra_img_name_list = []
        self.tra_lbl_name_list = []

        for img, seg in zip(data["img"], data["seg"]):
            self.tra_img_name_list.append(data_root + img)
            self.tra_lbl_name_list.append(data_root + seg)



        # self.gt_path = join(data_root, "gts")
        # self.img_path = join(data_root, "imgs")
        # self.gt_path_files = sorted(
        #     glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        # )
        # self.gt_path_files = [
        #     file
        #     for file in self.gt_path_files
        #     if os.path.isfile(join(self.img_path, os.path.basename(file)))
        # ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.tra_img_name_list)}")

    def __len__(self):
        return len(self.tra_img_name_list)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = self.tra_img_name_list[index]
        mask_name = self.tra_lbl_name_list[index]

        image = Image.open(img_name)
        img_1024 = np.array(image)

        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        # assert (
        #     np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        # ), "image should be normalized to [0, 1]"

        mask_image = Image.open(mask_name)
        gt = np.array(mask_image)
        # gt = np.load(
        #     self.gt_path_files[index], "r", allow_pickle=True
        # )  # multiple labels [0, 1,4,5...], (256,256)
        # assert img_name == os.path.basename(self.gt_path_files[index]), (
        #     "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        # )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )