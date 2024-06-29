# 导入了一些库
import os
import warnings

import torchvision
from torch.nn.functional import threshold, normalize
import cv2
import logging
from segment_anything_u2net.build_u2net_sam import build_sam
from utils.data_convert import build_dataloader_box
warnings.filterwarnings(action='ignore')
import numpy as np
import argparse
import torch
from torch.nn import functional as F
from skimage import io
from PIL import Image

# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ISBI', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help='')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=30, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models_box/', help='model path directory')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    return parser.parse_known_args()[0]

def create_clear_dir(dir):
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    else:
        file_list = os.listdir(dir)
        # 遍历文件列表并删除每个文件
        for file in file_list:
            # 构建完整的文件路径
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                # 如果是文件则直接删除
                os.remove(file_path)

def dice_iou(pred, target, smooth=1.0):
    # 读取并转换图像为二值化形式
    image1 = cv2.imread(pred, 0)

    image2 = cv2.imread(target, 0)
    _, image1_binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2_binary = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
    if image1_binary.shape != image2_binary.shape:
        raise ValueError("image1_binary.shape != image2_binary.shape")
    # 计算交集和并集
    intersection = cv2.bitwise_and(image1_binary, image2_binary)
    union = cv2.addWeighted(image1_binary, 0.5, image2_binary, 0.5, 0)

    # 计算DICE系数
    num_pixels_intersecting = cv2.countNonZero(intersection)
    num_pixels_total = cv2.countNonZero(union)
    dice_coefficient = (2 * num_pixels_intersecting+smooth) / float(num_pixels_total + num_pixels_intersecting+smooth)
    iou_coefficient = (num_pixels_intersecting + smooth) / float(num_pixels_total+smooth)
    return dice_coefficient, iou_coefficient

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

    model_path = "./models_box/"
    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"

    dataset_name = opt.dataset_name
    logging.basicConfig(filename=f'./val/{dataset_name }_val.log', encoding='utf-8', level=logging.DEBUG)

    sam = build_sam(checkpoint=checkpoint)
    sam = sam.to(device=device)
    sam.eval()

    print('val Start')
    dataloaders = build_dataloader_box(sam, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers)


    interaction_dir = f'./val/{opt.dataset_name}'
    create_clear_dir(interaction_dir)

    interaction_total_dice = 0
    interaction_total_iou = 0
    # 循环进行模型的多轮训练
    for index, val_data in enumerate(dataloaders['val']):
        # 将训练数据移到指定设备，这里是GPU
        val_input = val_data['image'].to(device)

        val_target_mask = val_data['mask'].to(device, dtype=torch.float32)
        prompt_box = val_data["prompt_box"].to(device)
        prompt_masks = val_data["prompt_masks"].to(device)
        mask_ratio_masks = val_data["mask_ratio_masks"].to(device)
        image_path = val_data['image_path'][0]
        mask_path = val_data['mask_path'][0]

        print("image_path:", image_path)
        logging.info("image_path:{}".format(image_path))
        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        # 使用 sam 模型的 image_encoder 提取图像特征，并使用 prompt_encoder 提取稀疏和密集的嵌入。在本代码中进行提示输入，所以都是None.
        val_encode_feature = sam.image_encoder(val_input)
        val_sparse_embeddings, val_dense_embeddings = sam.prompt_encoder(points=None, boxes=prompt_box,
                                                                         masks=prompt_masks)


        #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
        val_mask, val_IOU = sam.mask_decoder(
            image_embeddings=val_encode_feature,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=val_sparse_embeddings,
            dense_prompt_embeddings=val_dense_embeddings,
            multimask_output=False)

        H, W = val_target_mask.shape[-2:]
        low_res_pred = torch.sigmoid(val_mask)
        low_res = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        low_res = low_res * torch.where(mask_ratio_masks > 0, 1, 0)
        low_res = low_res.squeeze().cpu()
        res = torch.where(low_res > 0.5, 255.0, 0.0)

        if "\\" in image_path:
            aaa = image_path.split("\\")
        else:
            aaa = image_path.split("/")
        image_path = interaction_dir + '/' + aaa[len(aaa) - 1]
        torchvision.utils.save_image(res, image_path)

        interaction_dice, interaction_iou = dice_iou(image_path, mask_path)
        interaction_total_dice += interaction_dice
        interaction_total_iou += interaction_iou

        print("interaction iou:{:.6f}, interaction dice:{:.6f}".format(interaction_iou, interaction_dice))
        print("interaction mean iou:{:.6f},interaction mean dice:{:.6f}"
              .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))
        logging.info("interaction iou:{:.6f}, interaction dice:{:.6f}".format(interaction_iou, interaction_dice))
        logging.info("interaction mean iou:{:.6f},interaction mean dice:{:.6f}"
                     .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))




if __name__ == '__main__':
    opt = parse_opt()
    main(opt)