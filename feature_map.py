
import torchvision.transforms as transforms
import torch.nn as nn
from matplotlib import pyplot as plt

from BPAT_UNet.our_model.BPATUNet_all import BPATUNet
from MedSAM import MedSAM
from MedSAM_box import MedSAMBox

from MySegmentModel.modeling.MySegmentModel import build_model
from segment_anything.utils.transforms import ResizeLongestSide
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from utils.data_convert import getDatasets, build_dataloader
from torchvision import transforms
from PIL import Image

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
from torchcam.utils import overlay_mask
from pytorch_grad_cam import GradCAM

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='Thyroid_tn3k', help="dataset name")
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=True, help='is use box')
    parser.add_argument('-auxiliary_model_path', type=str, default='./BPAT_UNet/BPAT-UNet_best.pth')
    return parser

def main():
    opt = get_argparser().parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    auxiliary_model = BPATUNet(n_classes=1)
    auxiliary_model.load_state_dict(torch.load(opt.auxiliary_model_path, weights_only=True))
    auxiliary_model = auxiliary_model.to(device)
    auxiliary_model.eval()

    img_name_list, lbl_name_list,auxiliary_list = getDatasets(opt.dataset_name, opt.data_dir, "train")


    print('device', device)


    # 3-图像预处理
    # 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
    test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                         ])
    # 4-载入图片
    img_path = img_name_list[0]
    img_pil = Image.open(img_path)
    img_pil = img_pil.convert("RGB")
    input_tensor = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    print(input_tensor.shape)

    # 选择可解释性方法
    # GradCAM
    # target_layers = [model.basic_block1[-1],model.basic_block2[-1]] # 要分析的层
    target_layers = [auxiliary_model]
    # targets = [ClassifierOutputTarget(14)] # 要分析的类别
    targets = None
    cam = GradCAM(model=auxiliary_model, target_layers=target_layers)
    # 生成Grad-CAM热力图
    cam_map = cam(input_tensor=input_tensor, targets=targets)[0]  # 不加平滑
    # cam_map = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0] # 加平滑
    print(cam_map.shape)
    plt.imshow(cam_map)
    plt.title('Grad-CAM')
    plt.show()

    # 叠加在原图像
    result = overlay_mask(img_pil, Image.fromarray(cam_map), alpha=0.5)  # alpha越小，原图越淡
    plt.imshow(result)
    plt.title('Grad-CAM')
    plt.show()


if __name__ == "__main__":
    main()
