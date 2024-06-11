
import torchvision.transforms as transforms
import torch.nn as nn
from MedSAM import MedSAM
from U2_Net.data_loader import RescaleT1, ToTensorLab1
from U2_Net.model import U2NET
from segment_anything.utils.transforms import ResizeLongestSide
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from utils.data_convert import getDatasets

device = torch.device("cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"


class U2NetTarget(nn.Module):
    def __init__(self,input):
        super(U2NetTarget,self).__init__()

        self.input = input

    def forward(self,x):
        # 读取图片，将图片转为RGB
        origin_img = cv2.imread(self.input)
        rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)


        # 3.图片预处理：resize、裁剪、归一化
        trans = transforms.Compose([RescaleT1(320)])
        crop_img = trans(rgb_img)

        label = crop_img[:, :, np.newaxis]
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)
        tmpLbl = np.zeros(label.shape)
        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpLbl = label.transpose((2, 0, 1))

        tmpLbl = np.ascontiguousarray(tmpLbl)
        crop_img = torch.from_numpy(tmpLbl)
        labels = crop_img.type(torch.FloatTensor).unsqueeze(0)
        bce_loss = nn.BCELoss(reduction='mean')
        loss = bce_loss(x, labels)
        return loss


class SAMTarget(nn.Module):
    def __init__(self,input):
        super(SAMTarget, self).__init__()

        self.input = input

    def forward(self,x):
        # 读取图片，将图片转为RGB
        origin_img = cv2.imread(self.input)
        rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)


        # 3.图片预处理：resize、裁剪、归一化
        trans = transforms.Compose([RescaleT1(320)])
        crop_img = trans(rgb_img)

        label = crop_img[:, :, np.newaxis]
        if (np.max(label) < 1e-6):
            label = label
        else:
            label = label / np.max(label)
        tmpLbl = np.zeros(label.shape)
        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpLbl = label.transpose((2, 0, 1))

        tmpLbl = np.ascontiguousarray(tmpLbl)
        crop_img = torch.from_numpy(tmpLbl)
        labels = crop_img.type(torch.FloatTensor).unsqueeze(0)
        bce_loss = nn.BCELoss(reduction='mean')
        loss = bce_loss(x, labels)
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
    # sam.eval()

    medsam = MedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)
    # mask_decoder = sam.mask_decoder
    medsam.eval()
    # print(medsam.mask_decoder.iou_prediction_head.layers[2])

    traget_layers = [medsam.mask_decoder.iou_prediction_head.layers[2]]

    print(traget_layers)
    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")
    index = 1
    # 读取图片，将图片转为RGB
    origin_img = cv2.imread(img_name_list[index])

    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    # 图片预处理：resize、裁剪、归一化
    transform = ResizeLongestSide(1024)
    img = transform.apply_image(rgb_img)  #
    img = torch.as_tensor(img)  # torch tensor 变更
    img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None
    crop_img = sam.preprocess(img.to(device=device))  # img nomalize or padding
    net_input = crop_img.unsqueeze(0)
    canvas_img = net_input.numpy()

    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)
    # 实例化cam，得到指定feature map的可视化数据
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=medsam, target_layers=traget_layers)
    grayscale_cam = cam(net_input, targets=[SAMTarget(lbl_name_list[index])])
    grayscale_cam = grayscale_cam[0, :]

    # 将feature map与原图叠加并可视化
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)

def main1():
    opt = get_argparser().parse_args()

    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")
    index = 0
    # 1.定义模型结构，选取要可视化的层
    # resnet18 = models.resnet18(pretrained=True)
    # resnet18.eval()
    net = U2NET(3, 1)
    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + opt.dataset_name + '.pth'
    net.load_state_dict(torch.load(model_dir, map_location=device))


    # print(net.outconv)
    traget_layers = [net.outconv]

    # 读取图片，将图片转为RGB
    origin_img = cv2.imread(img_name_list[index])
    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    # 3.图片预处理：resize、裁剪、归一化
    trans = transforms.Compose([RescaleT1(320), ToTensorLab1(flag=0)])
    crop_img = trans(rgb_img)
    net_input = crop_img.unsqueeze(0)
    net_input = net_input.type(torch.FloatTensor).to(device)

    # 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
    canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
    canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

    # 5.实例化cam
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=net, target_layers=traget_layers)
    grayscale_cam = cam(net_input, targets=[U2NetTarget(lbl_name_list[index])])
    grayscale_cam = grayscale_cam[0, :]

    # 6.将feature map与原图叠加并可视化
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
    cv2.imshow('feature map', visualization_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main1()
