# 导入了一些库
import os
import warnings
import cv2
import numpy as np
import torchvision.utils
from PIL import Image
warnings.filterwarnings(action='ignore')
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from U2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from U2_Net.model import U2NET
from utils.data_convert import getDatasets, save_output, calculate_iou_dice, mean_iou
import argparse
import torch
import shutil

warnings.filterwarnings(action='ignore')


# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='Thyroid', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')

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

    datasets = opt.datasets
    image_list, mask_list, _ = getDatasets(datasets, opt.data_dir, "test")
    print("Number of images: ", len(image_list))

    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + datasets + '.pth'
    test_salobj_dataset = SalObjDataset(img_name_list=image_list,
                                        lbl_name_list=mask_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_loader = DataLoader(test_salobj_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=device))
    net.to(device)
    net.eval()
    total_dice = 0
    total_iou = 0
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            imidx = data["imidx"].item()

            image_path = image_list[imidx]
            mask_path = mask_list[imidx]
            inputs = data["image"]
            #####################################  U2Net
            inputs = inputs.type(torch.FloatTensor).to(device)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            arr = image_path.split("/")
            path = ""
            index1 = 0
            for var in arr:
                if len(var) > 0:
                    path += var + "/"
                if index1 == len(arr) - 3:
                    break
                index1 = index1 + 1
            path = path + "bbox/"
            if not os.path.exists(path):
                # 如果文件夹不存在，则创建文件夹
                os.makedirs(path)
            save_image_name = path + arr[len(arr) - 1]
            if os.path.exists(save_image_name):
                os.remove(save_image_name)
            image = Image.open(image_path)
            image_np = np.array(image)
            if len(image_np.shape) == 2:
                h, w = image_np.shape
            else:
                h, w, _ = image_np.shape

            pres = torch.where(d0.squeeze() > 0.5, 255.0, 0)
            predict_np = pres.cpu().data.numpy()
            im = Image.fromarray(predict_np).convert('L')
            imo = im.resize((w, h), resample=Image.BILINEAR)
            imo.save(save_image_name)

            u2net_iou, u2net_dice = calculate_iou_dice(save_image_name, mask_path)
            total_dice += u2net_dice.item()
            total_iou += u2net_iou.item()
            print("index:{}/{}, image:{}, u2net_dice:{}, u2net_iou:{}".
                  format(index, len(test_loader), image_path, u2net_dice.item(), u2net_iou.item()))

            if u2net_iou > 0.1:
                continue
            name = arr[len(arr) - 1]
            if not os.path.exists(opt.datasets):
                os.makedirs(opt.datasets)
            path = f'{opt.datasets}/{name}_{u2net_iou.item()}'
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copyfile(image_path, path + "/image.png")
            shutil.copyfile(mask_path, path + "/mask.png")
            shutil.copyfile(save_image_name, path + "/predict.png")

    print("mean iou:{:.6f}, mean dice:{:.6f}".format(total_iou/len(test_loader), total_dice/len(test_loader)))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)