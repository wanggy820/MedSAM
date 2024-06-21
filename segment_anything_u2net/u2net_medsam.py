import os
import cv2
import numpy as np
import torch
import torchvision
from skimage import transform
from torch import nn
import torch.nn.functional as F
join = os.path.join


class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 前向传播计算
        output = (input > 0.5).float()
        # 保存输入张量，以便在反向传播中使用
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播计算梯度
        grad_input = (grad_output > 0.5).float()
        return grad_input

class U2NetRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.ratio = 1.0

    def forward(self, x, mask_path):
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        H, W = mask_np.shape
        h = int(H * self.ratio)
        w = int(W * self.ratio)
        mask_ratio = transform.resize(
            mask_np, (h, w), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        top = int(h * (self.ratio - 1) / 2)
        bottom = top + H
        left = int(w * (self.ratio - 1) / 2)
        right = left + W
        mask_ratio_np = mask_ratio[top:bottom, left:right]
        mask_ratio_np += mask_np
        mask_ratio_np = mask_ratio_np // 2 + mask_ratio_np % 2
        mask_ratio_masks = torch.as_tensor(mask_ratio_np, dtype=torch.float32)  # torch tensor
        mask_ratio_masks = mask_ratio_masks.unsqueeze(0).unsqueeze(0)
        return x * torch.where(mask_ratio_masks > 0, 1, 0)

class U2NetMedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        res,
    ):
        super().__init__()
        self.boxes = None
        self.masks = None
        self.mask_path = None
        self.width = 0
        self.height = 0
        self.ratio = 1
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.res = res
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def setBox(self, boxes, masks, mask_path, width, height):
        self.boxes = boxes
        self.masks = masks
        self.mask_path = mask_path
        self.width = width
        self.height = height

    def forward(self, image, boxes=None, masks=None):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        mask_np = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        H, W = mask_np.shape
        h = int(H * self.ratio)
        w = int(W * self.ratio)
        mask_ratio = transform.resize(
            mask_np, (h, w), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        top = int(h * (self.ratio - 1) / 2)
        bottom = top + H
        left = int(w * (self.ratio - 1) / 2)
        right = left + W
        mask_ratio_np = mask_ratio[top:bottom, left:right]
        mask_ratio_np += mask_np
        mask_ratio_np = mask_ratio_np // 2 + mask_ratio_np % 2
        mask_ratio_masks = torch.as_tensor(mask_ratio_np, dtype=torch.float32)  # torch tensor
        mask_ratio_masks = mask_ratio_masks.unsqueeze(0).unsqueeze(0)

        if boxes is None:
            boxes = self.boxes
        if boxes is None:
            boxes_torch = None
        else:
            boxes_torch = torch.as_tensor(boxes, dtype=torch.float32, device=image.device)
            if len(boxes_torch.shape) == 2:
                boxes_torch = boxes_torch[:, None, :]  # (B, 1, 4)

        if masks is None:
            masks = self.masks
        if masks is None:
            masks_torch = None
        else:
            masks_torch = torch.as_tensor(masks, dtype=torch.float, device=image.device)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=boxes_torch,
                masks=masks_torch,
            )

        low_res_masks = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        low_res_pred = torch.sigmoid(low_res_masks)  # (1, 1, 256, 256)
        low_res = F.interpolate(
            low_res_pred,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        res = self.res(low_res, self.mask_path)
        # res = low_res * torch.where(mask_ratio_masks > 0, 1, 0)
        res = res.cpu()

        # c2 = res.squeeze().cpu()
        # c3 = torch.where(c2 > 0.5, 255.0, 0.0)
        # torchvision.utils.save_image(c3, "res.png")
        #
        #
        # c2 = mask_ratio_masks.squeeze().cpu()
        # c3 = torch.where(c2 > 0.5, 255.0, 0.0)
        # torchvision.utils.save_image(c3, "mask_ratio_masks.png")
        #
        #
        # c2 = low_res.squeeze().cpu()
        # c3 = torch.where(c2 > 0.5, 255.0, 0.0)
        # torchvision.utils.save_image(c3, "low_res.png")

        output = MyFunction.apply(res)
        return output
