import argparse
import os
import cv2
import numpy as np
import torch
from skimage import transform, io
from segment_anything import sam_model_registry
from torch.nn import functional as F
import logging

from utils.box import find_bboxes
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

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='ISBI', help="dataset name")
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=False, help='is use box')
    return parser
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

colors = [
    (255, 255, 255),
]

@torch.no_grad()
def medsam_inference(sam, img_embed, prompt_box, prompt_masks, height, width):
    box_torch = torch.as_tensor(prompt_box, dtype=torch.float, device=img_embed.device)
    if prompt_masks is None:
        masks_torch = None
    else:
        masks_torch = torch.as_tensor(prompt_masks, dtype=torch.float, device=img_embed.device)

    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=masks_torch,
    )
    low_res_logits, _ = sam.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
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

    res_pred = low_res_pred
    if len(low_res_pred.shape) == 3:
        medsam_seg = 0
        for i in range(0, low_res_pred.shape[0]):
            medsam_seg += low_res_pred[i]
        res_pred = medsam_seg
    medsam_seg = (res_pred > 0.5).astype(np.uint8)
    return medsam_seg

@torch.no_grad()
def get_embeddings(sam, img_3c):
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
        embedding = sam.image_encoder(
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
def interaction_u2net_predict(sam, image_path, mask_path, user_box, save_dir):
    img_np = io.imread(image_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    embedding = get_embeddings(sam, img_3c)
    H, W, _ = img_3c.shape

    mask_np = io.imread(mask_path)
    bboxes = find_bboxes(mask_np)
    # print("bounding box:", box_np)
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

    sam_mask = medsam_inference(sam, embedding, prompt_box, prompt_masks, H, W)

    mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")
    mask_c[sam_mask != 0] = colors[0]

    aaa = image_path.split("/")
    image_path = save_dir + '/' + aaa[len(aaa) - 1]
    io.imsave(image_path, mask_c)
    return image_path

def main():
    opt = get_argparser().parse_args()
    interaction_dir = f'./val/{opt.dataset_name}_{opt.use_box}'
    create_clear_dir(interaction_dir)

    # set up model
    model_path = "./models_no_box/"
    if opt.use_box:
        model_path = "./models_box/"
    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"
    if not os.path.exists(checkpoint):
        checkpoint = './work_dir/MedSAM/medsam_vit_b.pth'

    checkpoint = './work_dir/MedSAM/medsam_vit_b.pth'
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint).to(device)
    sam.eval()

    img_name_list, lbl_name_list = getDatasets(opt.dataset_name, opt.data_dir, "val")
    print("Number of images: ", len(img_name_list))

    dataset_name = opt.dataset_name
    logging.basicConfig(filename=f'./val/{dataset_name }_{opt.use_box}_val.log', encoding='utf-8', level=logging.DEBUG)

    with torch.no_grad():
        # --------- 4. inference for each image ---------
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, (image_path, mask_path) in enumerate(zip(img_name_list, lbl_name_list)):
            inferencing = image_path.split(os.sep)[-1]
            print("inferencing:", inferencing)
            logging.info("inferencing:{}".format(inferencing))

            # 还可以优化, 不保存图片，直接计算 dice, iou
            interaction_image_path = interaction_u2net_predict(sam, image_path, mask_path, opt.use_box, interaction_dir)
            interaction_dice, interaction_iou = dice_iou(interaction_image_path, lbl_name_list[index])
            interaction_total_dice += interaction_dice
            interaction_total_iou += interaction_iou

            print("interaction iou:{:.6f}, interaction dice:{:.6f}".format(interaction_iou, interaction_dice))
            print("interaction mean iou:{:.6f},interaction mean dice:{:.6f}"
                  .format(interaction_total_iou / (index + 1),interaction_total_dice / (index + 1)))
            logging.info("interaction iou:{:.6f}, interaction dice:{:.6f}".format(interaction_iou, interaction_dice))
            logging.info("interaction mean iou:{:.6f},interaction mean dice:{:.6f}"
                         .format(interaction_total_iou / (index + 1),interaction_total_dice / (index + 1)))


if __name__ == "__main__":
    main()
