import math
import os

import torch
from torch import nn
from .mask_encoder import MaskEncoder
from .pixel_encoder import PixelEncoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer
from TRFE_Net.model.unet import Unet
from utils.data_convert import get_click_prompt
from typing import Any
from torch.nn import functional as F

class MySegmentModel(nn.Module):
    def __init__(self,
                 backbone: Unet,
                 pixel_encoder: PixelEncoder,
                 mask_encoder: MaskEncoder,
                 prompt_encoder: PromptEncoder,
                 mask_decoder: MaskDecoder,
                 ) -> None:
        super().__init__()

        self.backbone = backbone
        self.pixel_encoder = pixel_encoder
        self.mask_encoder = mask_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        pixel_mean = [123.675, 116.28, 103.53],
        pixel_std = [58.395, 57.12, 57.375],
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    def forward(self, data):
        image = data['image_256'].to(self.device)
        prompt_box = data["prompt_box"].to(self.device)
        prompt_masks = data["prompt_masks"].to(self.device)
        points = get_click_prompt(data, self.device)

        x = self.backbone(image)
        x1 = x.sigmoid()
        x1 = self.pixel_encoder(x1)
        mask_features = F.interpolate(prompt_masks, size=self.pixel_encoder.image_size, mode="bilinear",
                                      align_corners=False)
        encoder_feature = self.mask_encoder(x1, mask_features)
        mask_features = F.interpolate(encoder_feature, size=self.prompt_encoder.embed_dim, mode="bilinear",
                                      align_corners=False)
        # (bs, 256, 64, 64)
        n, c, w, h = encoder_feature.shape
        w = int(math.sqrt(c * w * h // self.prompt_encoder.embed_dim))

        mask_former = encoder_feature.reshape(n, self.prompt_encoder.embed_dim, w, w)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=prompt_box, masks=prompt_masks)
        #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
        pre_mask, iou = self.mask_decoder(
            image_embeddings=mask_former,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        x = x.sigmoid()
        mask_features = mask_features.sigmoid()
        pre_mask = pre_mask.sigmoid()

        low_res_pred = x * 0.1 + mask_features * 0.2 + pre_mask * 0.4 + (prompt_masks/255.0) * 0.3
        return x, mask_features, low_res_pred


def build_model(checkout=None) -> MySegmentModel:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    mask_encoder_depth = 12
    pixel_encoder_embed_dim = 1280
    image_embedding_size = image_size // vit_patch_size

    backbone = Unet(3, 1)
    # state_dict = torch.load("../save_models/Thyroid_tn3k_fold0_unet/vit_b_1.00/sam_best.pth", map_location="cpu", weights_only=False)
    # backbone.load_state_dict(state_dict)
    pixel_encoder = PixelEncoder(img_size=image_size, patch_size=vit_patch_size, embed_dim=pixel_encoder_embed_dim)
    mask_encoder = MaskEncoder(depth=mask_encoder_depth)

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=4,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=5,
        iou_head_hidden_dim=256,
    )

    model = MySegmentModel(backbone, pixel_encoder, mask_encoder, prompt_encoder, mask_decoder)
    if checkout is not None and os.path.exists(checkout):
        model.load_state_dict(torch.load(checkout, map_location="cpu", weights_only=False))
    return model