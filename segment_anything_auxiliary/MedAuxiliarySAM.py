import numpy as np
import torch
from torch import nn
import random

bce_loss = nn.BCELoss(reduction='mean')
def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

class MedAuxiliarySAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        u2net,
        device,
        bbox_shift=0,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.u2net = u2net
        self.device = device
        self.bbox_shift = bbox_shift
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, auxiliary_image, user_box=True):
        auxiliary_image = auxiliary_image.to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.u2net(auxiliary_image)

        prompt_masks = []
        prompt_boxes = []
        output_size = 1024
        for r in d0:
            r = r.squeeze()
            y_indices, x_indices = np.where(r.detach().cpu().numpy() > 0.5)
            if len(y_indices) == 0 or len(x_indices) == 0:
                x_min = y_min = 0
                x_max = y_max = output_size
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                x_max = min(output_size, x_max + random.randint(0, self.bbox_shift))
                y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                y_max = min(output_size, y_max + random.randint(0, self.bbox_shift))

            box_1024 = np.array([x_min, y_min, x_max, y_max])*4
            prompt_boxes.append(box_1024)
            bbox_shift = random.randint(0, self.bbox_shift * 2)
            ratio = min(bbox_shift / (x_max - x_min) + 1, 1.5)
            ratio = max(ratio, 1)
            size = int(output_size * ratio)

            auxiliary_256 = r.squeeze()
            image_256 = r.unsqueeze(0).unsqueeze(0)
            auxiliary_ratio = torch.nn.functional.interpolate(image_256, scale_factor=ratio,
                                                        mode='bilinear',
                                                        align_corners=False)
            auxiliary_ratio = auxiliary_ratio.squeeze()

            top = int(size * (ratio - 1) / 2)
            bottom = top + output_size
            left = int(size * (ratio - 1) / 2)
            right = left + output_size
            auxiliary_ratio_masks = auxiliary_ratio[top:bottom, left:right]

            auxiliary_ratio_masks += auxiliary_256
            auxiliary_ratio_masks = auxiliary_ratio_masks // 2 + auxiliary_ratio_masks % 2
            prompt_mask = torch.where(auxiliary_ratio_masks > 0, 1.0, 0.0).unsqueeze(0)
            prompt_masks.append(prompt_mask)

        prompt_masks = torch.stack(prompt_masks).to(self.device)
        prompt_boxes = np.array(prompt_boxes)
        prompt_boxes = torch.from_numpy(prompt_boxes).to(self.device)
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            if user_box:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=prompt_boxes,
                    masks=None,
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=prompt_masks,
                )
        low_res_masks, iou = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        res_masks = torch.sigmoid(low_res_masks)
        if ((res_masks > 0.5).sum() < (d0 > 0.5).sum()):
            res_masks = d0

        return iou, res_masks, d0, d1, d2, d3, d4, d5, d6