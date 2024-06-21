import torch.nn as nn
from skimage import transform, io
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os
import cv2
import numpy as np
import torch

from U2_Net.model import U2NET
from segment_anything.modeling import TwoWayTransformer, PromptEncoder
from segment_anything.modeling.mask_decoder_feature_map import FeatureMapMaskDecoder
from segment_anything_u2net.u2net_encoder import U2NetEncoder
from segment_anything_u2net.u2net_medsam import U2NetMedSAM, U2NetRes
from segment_anything_u2net.u2net_sam import U2NetSam
from utils.box import find_bboxes
from utils.data_convert import getDatasets


device = torch.device("cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"

def interaction_u2net_predict(sam, mask_path):
    mask_np = io.imread(mask_path)
    H, W = mask_np.shape
    bboxes = find_bboxes(mask_np)
    prompt_box = bboxes / np.array([W, H, W, H]) * 1024

    mask_256 = transform.resize(
        mask_np, (256, 256), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    mask_256 = (mask_256 - mask_256.min()) / np.clip(
        mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 1)
    prompt_masks = np.expand_dims(mask_256, axis=0).astype(np.float32)

    sam.setBox(prompt_box, prompt_masks, mask_path, W, H)


def get_img_1024_tensor(img_3c):
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    return img_1024_tensor



class SAMTarget(nn.Module):
    def __init__(self, input):
        super(SAMTarget, self).__init__()

        self.input = input

    def forward(self, x):
        # 读取图片，将图片转为RGB
        origin_img = cv2.imread(self.input)
        rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

        prompt_masks = np.expand_dims(rgb_img, axis=0).astype(np.float32)
        crop_img = torch.from_numpy(prompt_masks) / 255

        bce_loss = nn.BCELoss(reduction='mean')
        loss = bce_loss(x, crop_img)
        return loss


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='MICCAI', help="dataset name")
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    return parser


def build_sam(checkpoint=None):
    in_ch = 3
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    u2net = U2NET(in_ch, prompt_embed_dim)
    sam = U2NetSam(
        image_encoder=U2NetEncoder(u2net, image_embedding_size, image_size),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=FeatureMapMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        sam.load_state_dict(state_dict)
    return sam

def main():
    opt = get_argparser().parse_args()
    # set up model
    model_path = "./segment_anything_u2net/models_box/"

    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"

    sam = build_sam(checkpoint=checkpoint).to(device)

    res = U2NetRes()
    medsam = U2NetMedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder, res)
    medsam.eval()

    print(medsam)
    target_layers = [medsam.res]

    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")

    # 实例化cam，得到指定feature map的可视化数据
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=medsam, target_layers=target_layers)

    for image_path, mask_path in zip(img_name_list, lbl_name_list):
        print(f"image_path:{image_path}")

        interaction_u2net_predict(medsam, mask_path)

        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape
        net_input = get_img_1024_tensor(img_3c)

        canvas_img = cv2.cvtColor(img_3c, cv2.COLOR_RGB2BGR)

        grayscale_cam = cam(net_input, targets=[SAMTarget(mask_path)])
        grayscale_cam = grayscale_cam[0, :]

        origin_cam = cv2.resize(grayscale_cam, (W, H))

        # 将feature map与原图叠加并可视化
        src_img = np.float32(canvas_img) / 255


        visualization_img = show_cam_on_image(src_img, origin_cam, use_rgb=False)

        arr = image_path.split("/")
        image_name = arr[len(arr) - 1]
        path = "./feature_map"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path += os.sep + opt.dataset_name
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path += os.sep + image_name

        io.imsave(path, visualization_img)

        # cv2.imshow('feature map', visualization_img)
        # cv2.waitKey(0)




if __name__ == "__main__":
    main()
