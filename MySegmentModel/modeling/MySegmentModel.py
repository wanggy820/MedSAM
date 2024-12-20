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
        x1 = self.pixel_encoder(x)
        mask_features = F.interpolate(prompt_masks, size=self.pixel_encoder.image_size, mode="bilinear",
                                      align_corners=False)
        encode_feature = self.mask_encoder(x1, mask_features)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=points, boxes=prompt_box, masks=prompt_masks)
        #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
        pre_mask, iou = self.mask_decoder(
            image_embeddings=encode_feature,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        low_res_pred = torch.sigmoid(pre_mask)
        return low_res_pred, iou


def build_model(checkout=None) -> MySegmentModel:
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    mask_encoder_depth = 2
    pixel_encoder_embed_dim = 768
    image_embedding_size = image_size // vit_patch_size

    backbone = Unet(3, 1)
    pixel_encoder = PixelEncoder(img_size=image_size, patch_size=vit_patch_size, embed_dim=pixel_encoder_embed_dim)
    mask_encoder = MaskEncoder(depth=mask_encoder_depth, prompt_embed_dim=prompt_embed_dim)

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
        model.load_state_dict(torch.load(checkout, map_location="cpu"))
    return model