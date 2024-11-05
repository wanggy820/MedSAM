import torch
from torch import nn

from utils.data_convert import get_click_prompt


class MySAMModel(nn.Module):
    def __init__( self, sam, auxiliary_model):
        super().__init__()
        self.sam = sam
        self.auxiliary_model = auxiliary_model
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, data):
        image = data['image'].to(self.sam.device)
        prompt_box = data["prompt_box"].to(self.sam.device)
        prompt_masks = data["prompt_masks"].to(self.sam.device)
        points = get_click_prompt(data, self.sam.device)

        with torch.no_grad():
            encode_feature = self.sam.image_encoder(image)  # (3, 256, 64, 64)
            # 使用 sam 模型的 image_encoder 提取图像特征，并使用 prompt_encoder 提取稀疏和密集的嵌入。在本代码中进行提示输入，所以都是None.
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=points, boxes=prompt_box, masks=prompt_masks)
        #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
        pre_mask, iou = self.sam.mask_decoder(
            image_embeddings=encode_feature,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        low_res_pred = torch.sigmoid(pre_mask)
        return low_res_pred, iou