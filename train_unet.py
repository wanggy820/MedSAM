# 导入了一些库
import json
import warnings

import cv2
import torchvision.utils
import torch.nn as nn
from BPAT_UNet.our_model.BPATUNet_all import BPATUNet
from MySAMModel import MySAMModel
from TRFE_Net.model.unet import Unet
from TRFE_Net.model.utils import SoftDiceLoss
from utils.data_convert import mean_iou, compute_loss, build_dataloader, get_click_prompt
from torchvision.transforms import ToPILImage
warnings.filterwarnings(action='ignore')
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim
from segment_anything import sam_model_registry
import torch.nn.functional as F
from TRFE_Net.visualization.metrics import Metrics, evaluate
from torchvision.transforms import ToPILImage
# 设置了一些配置参数
beta = (0.9, 0.999)
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Thyroid_tn3k', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=90, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help='')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=100, help='train epcoh')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--save_models_path', type=str, default='./save_models', help='model path directory')
    parser.add_argument('--vit_type', type=str, default='vit_b', help='sam vit type')
    parser.add_argument('--ratio', type=float, default=1.00, help='ratio')
    parser.add_argument('-fold', type=int, default=0)
    parser.add_argument('-auxiliary_model_path', type=str, default='./BPAT_UNet/BPAT-UNet_best.pth')
    return parser.parse_known_args()[0]

def main(opt):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(234)
    if device == f'cuda:0':
        torch.cuda.manual_seed_all(234)
    #  脚本使用预先构建的架构（sam_model_registry['vit_b']）定义了一个神经网络模型，并设置了优化器（AdamW）和学习率调度。
    print(device, 'is available')
    print("Loading model...")
    epoch_add = 0
    lr = opt.lr

    save_models_path = opt.save_models_path
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)
    dataset_model = f"{save_models_path}/{opt.dataset_name}_fold{opt.fold}_unet"
    if not os.path.exists(dataset_model):
        os.makedirs(dataset_model)
    prefix = f"{dataset_model}/{opt.vit_type}_{opt.ratio:.2f}"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    best_checkpoint = f"{prefix}/sam_best.pth"

    tr_pl_loss_list = []
    tr_pl_miou_list = []
    tr_pl_dice_list = []

    best_mIOU = 0
    best_dice = 0
    start = 0
    if opt.vit_type == "vit_b":
        checkpoint = f'./work_dir/SAM/sam_vit_b_01ec64.pth'
    elif opt.vit_type == "vit_h":
        checkpoint = f'./work_dir/SAM/sam_vit_h_4b8939.pth'
    else:
        checkpoint = f'./work_dir/SAM/sam_vit_l_0b3195.pth'
    sam = sam_model_registry[opt.vit_type](checkpoint=checkpoint)
    sam = sam.to(device=device)

    net = Unet(in_ch=3, out_ch=1).to(device=device)
    net.load_state_dict(torch.load("./save_models/Thyroid_tn3k_fold0_unet/vit_b_1.00/sam_best.pth", map_location=torch.device("cpu")))
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=beta, weight_decay=opt.weight_decay)

    print('Training Start')
    net.train()
    auxiliary_model = BPATUNet(n_classes=1)
    auxiliary_model.load_state_dict(torch.load(opt.auxiliary_model_path, map_location=torch.device('cpu')))
    auxiliary_model = auxiliary_model.to(device)
    auxiliary_model.eval()
    dataloaders = build_dataloader(sam, auxiliary_model, opt.dataset_name, opt.data_dir, opt.batch_size,
                                   opt.num_workers, opt.ratio, opt.fold)

    best_loss = 99999
    cross_entropy_loss = SoftDiceLoss()
    for epoch in range(start, opt.epochs):
        train_loss_list = []

        # -------------- train --------------

        iterations = tqdm(dataloaders['train'])

        # 循环进行模型的多轮训练
        for train_data in iterations:
            edges = train_data['mask'].to(device, dtype=torch.float32)
            image_256 = train_data['image_256'].to(device, dtype=torch.float32)
            # 对优化器的梯度进行归零
            optimizer.zero_grad()

            low_res_pred = net(image_256)


            train_loss_one = cross_entropy_loss(low_res_pred, edges)
            train_loss_one.backward()

            optimizer.step()
            train_loss_list.append(train_loss_one.item())

            pbar_desc = "Model train loss --- "
            pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
            iterations.set_description(pbar_desc)

            # pre = torch.sigmoid(low_res_pred)
            # pre = torch.where(edges > 0.5, 1.0, 0.0)
            # c = pre.max()
            # b = pre.sum()
            # a= (pre*255).squeeze()
            # to_pil_image = ToPILImage()
            # pil_image = to_pil_image(a)
            # pil_image.save('image.png')
            # print('---')

        train_loss = np.mean(train_loss_list)

        torch.cuda.empty_cache()
        tr_pl_loss_list.append(train_loss)


        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), best_checkpoint)

        print("train epoch:{:3d}, best_loss:{:3.4f}, train_loss:{:3.4f}"
              .format(epoch + 1 + epoch_add, best_loss,train_loss))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)