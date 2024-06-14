import torch.nn as nn
from MedSAM import MedSAM
from skimage import transform, io
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from utils.data_convert import getDatasets, find_bboxes
from PIL import Image

device = torch.device("cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"


def interaction_u2net_predict(sam, mask_path, user_box):
    mask_np = io.imread(mask_path)
    H, W = mask_np.shape
    bboxes = find_bboxes(mask_np)
    prompt_box = bboxes / np.array([W, H, W, H]) * 1024

    prompt_masks = None
    if user_box:
        mask_256 = transform.resize(
            mask_np, (256, 256), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        mask_256 = (mask_256 - mask_256.min()) / np.clip(
            mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 1)
        prompt_masks = np.expand_dims(mask_256, axis=0).astype(np.float32)

    sam.setBox(prompt_box, prompt_masks, W, H)


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
        crop_img = torch.from_numpy(prompt_masks)/255

        bce_loss = nn.BCELoss(reduction='mean')
        loss = bce_loss(x, crop_img)
        return loss


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='MICCAI', help="dataset name")
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=True, help='is use box')
    return parser


def main():
    opt = get_argparser().parse_args()
    # set up model
    model_path = "./models_no_box/"
    if opt.use_box:
        model_path = "./models_box/"
    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"
    if not os.path.exists(checkpoint):
        checkpoint = './work_dir/SAM/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint).to(device)
    medsam = MedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)
    medsam.eval()

    # print(medsam.mask_decoder)
    target_layers = [medsam.image_encoder.neck[3]]

    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")
    index = 1
    image_path = img_name_list[index]
    mask_path = lbl_name_list[index]

    interaction_u2net_predict(medsam, mask_path, opt.use_box)

    img_np = io.imread(image_path)
    H, W, _ = img_np.shape
    net_input = get_img_1024_tensor(img_np)

    canvas_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 实例化cam，得到指定feature map的可视化数据
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=medsam, target_layers=target_layers)
    grayscale_cam = cam(net_input, targets=[SAMTarget(mask_path)])
    grayscale_cam = grayscale_cam[0, :]

    origin_cam = cv2.resize(grayscale_cam, (W, H))

    # 将feature map与原图叠加并可视化
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, origin_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
