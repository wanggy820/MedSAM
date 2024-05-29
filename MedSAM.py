import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
join = os.path.join
from skimage import transform

class MedSAM(nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, prompt_box, prompt_mask, height, width):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=prompt_box,
                masks=prompt_mask,
            )
        low_res_masks, iou = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(height[0], width[0]),
            mode="bilinear",
            align_corners=False,
        )
        low_res_pred = ori_res_masks.squeeze(1).detach().cpu().numpy()  # (256, 256)

        # medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
        # normalized_data = (medsam_seg * 255 / np.max(medsam_seg)).astype(np.uint8)

        # # cv2.imwrite('tt.png', normalized_data)
        # pre_mask = torch.tensor(normalized_data).float()
        return low_res_pred, iou

