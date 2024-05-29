import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from MedSAM_Dataset import MedSAM_Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from skimage import io
from PIL import Image

# 损失函数
def focal_loss(pred, target, gamma=2.0, alpha=0.25, reduction='mean'):
    # pred = F.sigmoid(pred)
    pt = torch.where(target == 1, pred, 1 - pred)
    ce_loss = F.binary_cross_entropy(pred, target, reduction="none")
    focal_term = (1 - pt).pow(gamma)
    loss = alpha * focal_term * ce_loss

    return loss.mean()


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()

    return 1 - ((2. * intersection + smooth) /
                (pred_flat.sum() + target_flat.sum() + smooth))

def dice_function(pred, target, smooth=1.0):
    pred = F.sigmoid(pred).squeeze(1).to(dtype=torch.float32)
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()

    return ((2. * intersection + smooth) /
                (pred_flat.sum() + target_flat.sum() + smooth))
def compute_loss(pred_mask, true_mask, pred_iou, true_iou):
    pred_mask = pred_mask/255
    true_mask = true_mask/255
    fl = focal_loss(pred_mask, true_mask)
    dl = dice_loss(pred_mask, true_mask)
    mask_loss = 20 * fl + dl
    iou_loss = F.mse_loss(pred_iou, true_iou)
    total_loss = mask_loss + iou_loss

    return total_loss


def mean_iou(preds, labels, eps=1e-6):
    pred_cls = (preds == 1).float()
    label_cls = (labels == 1).float()
    intersection = (pred_cls * label_cls).sum(1).sum(1)
    union = (1 - (1 - pred_cls) * (1 - label_cls)).sum(1).sum(1)
    intersection = intersection + (union == 0)
    union = union + (union == 0)
    ious = intersection / union

    return ious


def compute_sam_dice(sam, image_path, mask_path, bboxes):
    img_np = io.imread(image_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    box_1024 = bboxes / np.array([W, H, W, H]) * 1024

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=sam.device)

    datasets = MedSAM_Dataset(sam, [image_path], [mask_path])
    dataloaders = DataLoader(datasets, batch_size=1, shuffle=False,  num_workers=0, pin_memory=False)
    torch.cuda.empty_cache()

    sam.eval()
    with torch.no_grad():
        valid_loss_list = []
        valid_miou_list = []
        valid_dice_list = []

        for valid_data in dataloaders:
            img, mask = valid_data["image"], valid_data["mask"]
            valid_input = img.to(sam.device)
            valid_target_mask = mask.to(sam.device, dtype=torch.float32)

            valid_encode_feature = sam.image_encoder(valid_input)
            valid_sparse_embeddings, valid_dense_embeddings = sam.prompt_encoder(points=None, boxes=box_torch,
                                                                                 masks=None)

            valid_mask, valid_IOU = sam.mask_decoder(image_embeddings=valid_encode_feature,

                                                     image_pe=sam.prompt_encoder.get_dense_pe(),

                                                     sparse_prompt_embeddings=valid_sparse_embeddings,

                                                     dense_prompt_embeddings=valid_dense_embeddings,

                                                     multimask_output=False)

            valid_true_iou = mean_iou(valid_mask, valid_target_mask, eps=1e-6)
            valid_miou_list = valid_miou_list + valid_true_iou.tolist()

            valid_loss_one = compute_loss(valid_mask, valid_target_mask, valid_IOU, valid_true_iou)
            valid_loss_list.append(valid_loss_one.item())

            valid_dice_one = dice_function(valid_mask, valid_target_mask)
            valid_dice_list.append(valid_dice_one.item())

        valid_loss = np.mean(valid_loss_list)
        valid_miou = np.mean(valid_miou_list)
        valid_dice = np.mean(valid_dice_list)
        return valid_loss, valid_miou, valid_dice

def getDatasets(datasets, root_dir, data_type):
    if datasets == "ISBI":
        data_dir = root_dir + "ISBI/"
        if data_type == "train":
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
        if data_type == "test":
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"

        f = open(filePath, encoding="utf-8")
        names = pd.read_csv(f)
        image_list = []
        mask_list = []
        for img, seg in zip(names["img"], names["seg"]):
            image_list.append(data_dir + img)
            mask_list.append(data_dir + seg)
        return image_list, mask_list

    if datasets == "MICCAI":
        if data_type == "train":
            data_dir = root_dir + "MICCAI2023/train"
        if data_type == "test":
            data_dir = root_dir + "MICCAI2023/val"

        image_list = sorted(glob.glob(data_dir + "/image/*"))
        mask_list = sorted(glob.glob(data_dir + "/mask/*"))
        return image_list, mask_list

    if datasets == "Thyroid":
        data_dir = root_dir + "Thyroid_Dataset/tg3k/"

        with open(data_dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            if data_type == "train":
                names = data["train"]
            if data_type == "test":
                names = data["val"]
        format = ".jpg"
        image_list = []
        mask_list = []
        for name in names:
            image_path = data_dir + "Thyroid-image/" + "{:04d}".format(name) + format
            mask_path = data_dir + "Thyroid-mask/" + "{:04d}".format(name) + format
            image_list.append(image_path)
            mask_list.append(mask_path)
        return image_list, mask_list

    if datasets == "DRIVE":
        if data_type == "train":
            data_dir = root_dir + "DRIVE/training/"
        if data_type == "test":
            data_dir = root_dir + "DRIVE/test/"
        image_list = sorted(glob.glob(data_dir + "/images/*"))
        mask_list = sorted(glob.glob(data_dir + "/mask_jpg/*"))
        return image_list, mask_list

# 数据加载
def build_dataloader(sam, model_name, data_dir, batch_size, num_workers):
    dataloaders = {}
    for key in ['train', 'test']:
        image_list, mask_list = getDatasets(model_name, data_dir, key)
        dataloaders[key] = DataLoader(
            MedSAM_Dataset(sam, image_list, mask_list),
            batch_size=batch_size,
            shuffle=True if key != 'test' else False,
            num_workers=num_workers,
            pin_memory=False
        )
    return dataloaders


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma-mi)/2.0, 1.0, 0)
    return dn
def find_u2net_bboxes(input, image_name):
    # normalization
    pred = input[:, 0, :, :]
    masks = normPRED(pred)

    predict = masks.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('L')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # imo.save("33.png")

    pred = np.array(imo)

    if len(image.shape) == 3:
        H, W, _ = image.shape
    else:
        H, W = image.shape
    y_indices, x_indices = np.where(pred > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    box_np = np.array([[x_min, y_min, x_max, y_max]])

    return box_np
