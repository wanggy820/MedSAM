import torch
from torch import nn
from skimage import transform

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
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

class MedAuxiliarySAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        u2net
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.u2net = u2net
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, epoch=5):
        auxiliary_image = transform.resize(image,
                                           (self.prompt_encoder.embed_dim, self.prompt_encoder.embed_dim),
                                           mode='constant', order=0, preserve_range=True)
        d0, d1, d2, d3, d4, d5, d6 = self.u2net(auxiliary_image)
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        res_masks = 0
        best_maks = d0
        best_iou = 0
        for i in range(0, epoch):
            res_masks += d0
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=d0,
                )
            low_res_masks, iou = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            res_masks = torch.sigmoid(low_res_masks)
            if best_iou < iou:
                best_iou = iou
                best_maks = res_masks

        if (best_maks.sum() < d0.sum()):
            best_maks = d0
        return best_iou, best_maks, d0, d1, d2, d3, d4, d5, d6