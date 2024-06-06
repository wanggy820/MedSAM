import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import glob

from torchvision.transforms import transforms

from U2_Net.data_loader import ToTensorLab, RescaleT
from U2_Net.data_loader import SalObjDataset
from U2_Net.model import U2NET # full size version 173.6 MB
from skimage import transform, io
from segment_anything import sam_model_registry
from torch.nn import functional as F
import logging

from utils.data_convert import getDatasets

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "./models_no_box/MICCAI_model_2_2_4/MICCAI_sam_best.pth"
MEDSAM_IMG_INPUT_SIZE = 1024
prediction_dir = './val/predict_u2net_results'
interaction_dir = './val/interaction_u2net_results'
# set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma - mi) / 2.0, 1.0, 0)
    return dn

def save_output(image_name, pred, d_dir):
    pred = normPRED(pred)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    img_name = image_name.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    image_path = d_dir+'/'+imidx+'.png'
    imo.save(image_path)
    return image_path

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

def dice_iou_function(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return dice, iou


colors = [
    (255, 255, 255),
]

def find_u2net_bboxes(input, image_name):
    # normalization
    pred = input[:, 0, :, :]
    masks = normPRED(pred)

    predict = masks.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pred = np.array(imo)
    gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    maxW = 0
    maxH = 0
    maxX = 0
    maxY = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x*h > maxW*maxH:
            maxX = x
            maxY = y
            maxW = w
            maxH = h

    return np.array([[maxX, maxY, maxX + maxW, maxY + maxH]])

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

@torch.no_grad()
def get_embeddings(img_3c):
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

    # if self.embedding is None:
    with torch.no_grad():
        embedding = medsam_model.image_encoder(
            img_1024_tensor
        )  # (1, 256, 64, 64)
        return embedding

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
def interaction_u2net_predict(bboxes, file_path, save_dir):
    img_np = io.imread(file_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    embedding = get_embeddings(img_3c)
    H, W, _ = img_3c.shape

    # print("bounding box:", box_np)
    box_1024 = bboxes / np.array([W, H, W, H]) * 1024

    sam_mask = medsam_inference(medsam_model, embedding, box_1024, H, W)

    mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")
    mask_c[sam_mask != 0] = colors[0]

    aaa = file_path.split("/")
    image_path = save_dir + '/' + aaa[len(aaa) - 1]
    io.imsave(image_path, mask_c)
    return image_path


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='MICCAI', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=True, help='is use box')
    return parser

def main():
    opt = get_argparser().parse_args()
    create_clear_dir(prediction_dir)
    create_clear_dir(interaction_dir)

    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")
    print("Number of images: ", len(img_name_list))

    dataset_name = opt.dataset_name
    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + dataset_name + '.pth'

    logging.basicConfig(filename="./val/" + dataset_name + '_val' + '.log', encoding='utf-8', level=logging.DEBUG)
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
    net.to(device)
    net.eval()
    with torch.no_grad():
        # --------- 4. inference for each image ---------
        u2net_total_dice = 0
        u2net_total_iou = 0
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, data in enumerate(test_salobj_dataloader):
            inferencing = img_name_list[index].split(os.sep)[-1]
            print("inferencing:", inferencing)
            logging.info("inferencing:{}".format(inferencing))

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor).to(device)
            d1, d2, d3, d4, d5, d6, d7 = net(inputs)

            u2net_dice, u2net_iou = dice_iou_function(d1.cpu().numpy(), labels.cpu().numpy())
            # save results to test_results folder
            file_path = img_name_list[index]

            # u2net 图片保存
            # u2net_image_path = save_output(file_path, d1, prediction_dir)
            # u2net_dice, u2net_iou = dice_iou(u2net_image_path, lbl_name_list[index])
            u2net_total_dice += u2net_dice
            u2net_total_iou += u2net_iou


            bboxes = find_u2net_bboxes(d1, img_name_list[index])

            # 还可以优化, 不保存图片，直接计算 dice, iou
            interaction_image_path = interaction_u2net_predict(bboxes, file_path, interaction_dir)
            interaction_dice, interaction_iou = dice_iou(interaction_image_path, lbl_name_list[index])
            interaction_total_dice += interaction_dice
            interaction_total_iou += interaction_iou

            print("u2net       iou:{}, u2net       dice:{}".format(u2net_iou, u2net_dice))
            print("interaction iou:{}, interaction dice:{}".format(interaction_iou, interaction_dice))
            print("u2net       average iou:{},u2net       average dice:{}".format(u2net_total_iou / (index + 1),
                                                                                  u2net_total_dice / (index + 1)))
            print("interaction average iou:{},interaction average dice:{}".format(interaction_total_iou / (index + 1),
                                                                             interaction_total_dice / (index + 1)))
            logging.info("u2net       iou:{}, u2net       dice:{}".format(u2net_iou, u2net_dice))
            logging.info("interaction iou:{}, interaction dice:{}".format(interaction_iou, interaction_dice))
            logging.info("u2net       average iou:{},u2net       average dice:{}".format(u2net_total_iou / (index + 1),
                                                                                         u2net_total_dice / (
                                                                                                     index + 1)))
            logging.info("interaction average iou:{},interaction average dice:{}".format(interaction_total_iou / (index + 1),
                                                                                  interaction_total_dice / (index + 1)))


if __name__ == "__main__":
    main()
