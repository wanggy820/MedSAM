import glob
import json
import os
import pandas as pd
import torch
import torch.nn.functional as F
from MedSAM_Dataset import MedSAM_Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import threshold, normalize
from skimage import io
from PIL import Image
from MedSAM_box import MedSAMBox
from torchvision import transforms

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
    pred_mask = F.sigmoid(pred_mask).squeeze(1).to(dtype=torch.float32)
    fl = focal_loss(pred_mask, true_mask)
    dl = dice_loss(pred_mask, true_mask)
    mask_loss = 20 * fl + dl
    iou_loss = F.mse_loss(pred_iou, true_iou)
    total_loss = mask_loss + iou_loss

    return total_loss


def mean_iou(preds, labels, eps=1e-6):
    preds = normalize(threshold(preds, 0.0, 0)).squeeze(1)
    pred_cls = (preds == 1).float()
    label_cls = (labels == 1).float()
    intersection = (pred_cls * label_cls).sum(1).sum(1)
    union = (1 - (1 - pred_cls) * (1 - label_cls)).sum(1).sum(1)
    intersection = intersection + (union == 0)
    union = union + (union == 0)
    ious = intersection / union

    return ious

def getDatasets(dataset_name, root_dir, data_type):
    if dataset_name == "ISBI":
        data_dir = root_dir + "ISBI/"
        if data_type == "train":
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
        else:
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"

        f = open(filePath, encoding="utf-8")
        names = pd.read_csv(f)
        image_list = []
        mask_list = []
        for img, seg in zip(names["img"], names["seg"]):
            image_list.append(data_dir + img)
            if data_type == "test":
                arr = img.split("/")
                mask_list.append(data_dir + "bbox/" + arr[len(arr) - 1])
            else:
                mask_list.append(data_dir + seg)
        return image_list, mask_list

    if dataset_name == "MICCAI":
        if data_type == "train":
            data_dir = root_dir + "MICCAI2023/train/"
        if data_type == "val":
            data_dir = root_dir + "MICCAI2023/val/"
        if data_type == "test":
            data_dir = root_dir + "MICCAI2023/"

        image_list = sorted(glob.glob(data_dir + "image/*"))
        if data_type == "test":
            mask_list = sorted(glob.glob(data_dir + "bbox/*"))
        else:
            mask_list = sorted(glob.glob(data_dir + "mask/*"))
        return image_list, mask_list

    if dataset_name == "Thyroid":
        data_dir = root_dir + "Thyroid_Dataset/tg3k/"

        with open(data_dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            if data_type == "train":
                names = data["train"]
            else:
                names = data["val"]
        format = ".jpg"
        image_list = []
        mask_list = []
        for name in names:
            image_path = data_dir + "Thyroid-image/" + "{:04d}".format(name) + format
            if data_type == "test":
                mask_path = data_dir + "bbox/" + "{:04d}".format(name) + format
            else:
                mask_path = data_dir + "Thyroid-mask/" + "{:04d}".format(name) + format
            image_list.append(image_path)
            mask_list.append(mask_path)
        return image_list, mask_list

    if dataset_name == "DRIVE":
        if data_type == "train":
            data_dir = root_dir + "DRIVE/training/"
        if data_type == "val":
            data_dir = root_dir + "DRIVE/test/"
        if data_type == "test":
            data_dir = root_dir + "DRIVE/bbox/"

        image_list = glob.glob(data_dir + "/images/*")
        if data_type == "test":
            mask_list = glob.glob(data_dir + "/bbox/*")
        else:
            mask_list = glob.glob(data_dir + "/mask/*")
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
            pin_memory=True
        )
    return dataloaders

def build_dataloader_box(sam, dataset_name, data_dir, batch_size, num_workers):
    dataloaders = {}
    for key in ['train', 'val', 'test']:
        image_list, mask_list = getDatasets(dataset_name, data_dir, key)
        datasets = MedSAMBox(sam, image_list, mask_list, bbox_shift=0 if key != 'train' else 20)
        dataloaders[key] = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=False if key != 'train' else True,
            num_workers=num_workers,
            pin_memory=False
        )
    return dataloaders

# 定义转换管道
transform = transforms.Compose([
    transforms.ToTensor(), # 转换为Tensor
])
def calculate_dice_iou(pred_path, mask_path, smooth = 1e-5):
    pre_img = Image.open(pred_path)
    pred = transform(pre_img)

    mask_img = Image.open(mask_path)
    mask = transform(mask_img)
    # 确保pred和target的大小一致
    assert pred.size() == mask.size(), "Size of predictions and targets must be the same"

    # 将pred和target转换为布尔型，即0和1，1代表前景，0代表背景
    pred_positives = (pred == 1)
    mask_positives = (mask == 1)

    # 计算交集
    intersection = (pred_positives * mask_positives).sum()

    # 计算并集
    union = (pred_positives + mask_positives).sum()

    dice = (2 * intersection + smooth) / (union + intersection + smooth)
    # 计算IoU
    iou = (intersection + smooth) / (union + smooth)  # 添加1e-6以避免除以零
    return dice, iou

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma-mi)/2.0, 1.0, 0)
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