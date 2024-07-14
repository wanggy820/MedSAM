import glob
import json
import os
import pandas as pd
import torch
import torch.nn.functional as F
from MedSAM_Dataset import MedSAM_Dataset
from torch.utils.data import DataLoader
from skimage import io
from PIL import Image
from MedSAM_box import MedSAMBox
from torchvision import transforms

from segment_anything_auxiliary.SAMDataset import SAMDataset


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
def compute_loss(pred_mask, true_mask):
    fl = focal_loss(pred_mask, true_mask)
    dl = dice_loss(pred_mask, true_mask)
    mask_loss = 20 * fl + dl

    return mask_loss


def mean_iou(preds, labels, eps=1e-6):
    preds = preds.squeeze(1)
    labels = labels.squeeze(1)
    pred_cls = (preds >= 0.5).float()
    label_cls = (labels >= 0.5).float()
    intersection = (pred_cls * label_cls).sum(1).sum(1)
    union = (1 - (1 - pred_cls) * (1 - label_cls)).sum(1).sum(1)
    intersection = intersection + (union == 0) + eps
    union = union + (union == 0) + eps
    ious = intersection / union
    dice = 2*intersection/ (union + intersection)
    return ious, dice

def getDatasets(dataset_name, root_dir, data_type):
    image_list = []
    mask_list = []
    auxiliary_list = []
    if dataset_name == "ISIC2016":
        data_dir = root_dir + "ISIC2016/"
        if data_type == "train":
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
        else:
            filePath = data_dir + "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"

        f = open(filePath, encoding="utf-8")
        names = pd.read_csv(f)
        for img, seg in zip(names["img"], names["seg"]):
            image_list.append(data_dir + img)
            mask_list.append(data_dir + seg)
            if data_type == "test":
                arr = img.split("/")
                auxiliary_list.append(data_dir + "bbox/" + arr[len(arr) - 1])
            else:
                auxiliary_list.append(data_dir + seg)
        return image_list, mask_list, auxiliary_list

    if dataset_name == "ISIC2017":
        data_dir = root_dir + "ISIC2017/"
        if data_type == "train":
            filePath = data_dir + "ISIC-2017_Training_Data_metadata.csv"
        elif data_type == "val":
            filePath = data_dir + "ISIC-2017_Validation_Data_metadata.csv"
        else:
            filePath = data_dir + "ISIC-2017_Test_v2_Data_metadata.csv"
        f = open(filePath, encoding="utf-8")
        names = pd.read_csv(f)
        for img in names["image_id"]:
            if data_type == "train":
                image_list.append(data_dir + "ISIC-2017_Training_Data/" + img + ".jpg")
                mask_list.append(data_dir + "ISIC-2017_Training_Part1_GroundTruth/" + img + "_segmentation.png")
                auxiliary_list.append(data_dir + "ISIC-2017_Training_Part1_GroundTruth/" + img + "_segmentation.png")
            elif data_type == "val":
                image_list.append(data_dir + "ISIC-2017_Validation_Data/" + img + ".jpg")
                mask_list.append(data_dir + "ISIC-2017_Validation_Part1_GroundTruth/" + img + "_segmentation.png")
                auxiliary_list.append(data_dir + "ISIC-2017_Validation_Part1_GroundTruth/" + img + "_segmentation.png")
            else:
                image_list.append(data_dir + "ISIC-2017_Test_v2_Data/" + img + ".jpg")
                mask_list.append(data_dir + "ISIC-2017_Test_v2_Part1_GroundTruth/" + img + "_segmentation.png")
                auxiliary_list.append(data_dir + "bbox/" + img + ".jpg")
        return image_list, mask_list, auxiliary_list

    if dataset_name == "ISIC2018":
        data_dir = root_dir + "ISIC2018/"

        if data_type == "train":
            image_list = sorted(glob.glob(data_dir + "ISIC2018_Task1-2_Training_Input/*.jpg"))
            mask_list = sorted(glob.glob(data_dir + "ISIC2018_Task1_Training_GroundTruth/*.png"))
            auxiliary_list = mask_list
        elif data_type == "val":
            image_list = sorted(glob.glob(data_dir + "ISIC2018_Task1-2_Validation_Input/*.jpg"))
            mask_list = sorted(glob.glob(data_dir + "ISIC2018_Task1_Validation_GroundTruth/*.png"))
            auxiliary_list = mask_list
        else:
            image_list = sorted(glob.glob(data_dir + "ISIC2018_Task1-2_Test_Input/*.jpg"))
            mask_list = sorted(glob.glob(data_dir + "ISIC2018_Task1_Test_GroundTruth/*.png"))
            auxiliary_list = sorted(glob.glob(data_dir + "bbox/*.png"))

        return image_list, mask_list, auxiliary_list

    if dataset_name == "MICCAI":
        if data_type == "train":
            data_dir = root_dir + "MICCAI2023/train/"
        else:
            data_dir = root_dir + "MICCAI2023/val/"

        image_list = sorted(glob.glob(data_dir + "image/*"))
        mask_list = sorted(glob.glob(data_dir + "mask/*"))
        if data_type == "test":
            auxiliary_list = sorted(glob.glob(data_dir + "bbox/*"))
        else:
            auxiliary_list = sorted(glob.glob(data_dir + "mask/*"))
        return image_list, mask_list, auxiliary_list

    if dataset_name == "Thyroid":
        data_dir = root_dir + "Thyroid_Dataset/tg3k/"

        with open(data_dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            if data_type == "test":
                names = data["val"]
            else:
                names = data[data_type]
        format = ".jpg"
        for name in names:
            image_path = data_dir + "Thyroid-image/" + "{:04d}".format(name) + format
            mask_path = data_dir + "Thyroid-mask/" + "{:04d}".format(name) + format
            auxiliary_path = data_dir + "bbox/" + "{:04d}".format(name) + format
            image_list.append(image_path)
            mask_list.append(mask_path)
            auxiliary_list.append(auxiliary_path)
        return image_list, mask_list, auxiliary_list

    if dataset_name == "DRIVE":
        if data_type == "train":
            data_dir = root_dir + "DRIVE/training/"
        else:
            data_dir = root_dir + "DRIVE/test/"

        image_list = glob.glob(data_dir + "/images/*")
        mask_list = glob.glob(data_dir + "/mask/*")
        if data_type == "test":
            auxiliary_list = glob.glob(root_dir + "DRIVE/bbox/*")
        else:
            auxiliary_list = mask_list
        return image_list, mask_list, auxiliary_list


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
        image_list, mask_list, auxiliary_list = getDatasets(dataset_name, data_dir, key)
        datasets = MedSAMBox(sam, image_list, mask_list, auxiliary_list, bbox_shift=20)
        dataloaders[key] = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=False if key != 'train' else True,
            num_workers=num_workers,
            pin_memory=False
        )
    return dataloaders

def build_dataloader_auxiliary(dataset_name, data_dir, batch_size, num_workers):
    dataloaders = {}
    for key in ['train', 'val', 'test']:
        image_list, mask_list, auxiliary_list = getDatasets(dataset_name, data_dir, key)
        datasets = SAMDataset(image_list, mask_list)
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
    transforms.ToTensor(),  # 转换为Tensor
])

def calculate_iou_dice1(pred, target, smooth=1e-6):
    pre_img = Image.open(pred)
    pred = transform(pre_img)

    mask_img = Image.open(target)
    mask = transform(mask_img)
    # 确保pred和target的大小一致
    assert pred.size() == mask.size(), "Size of predictions and targets must be the same"

    # 将pred和target转换为布尔型，即0和1，1代表前景，0代表背景
    pred_positives = (pred >= 0.5)
    pre_negatives = (pred < 0.5)
    mask_positives = (mask >= 0.5)
    mask_negatives = (mask < 0.5)

    TP = (pred_positives * mask_positives).sum()
    FP = (pred_positives * mask_negatives).sum()
    FN = (pre_negatives * mask_positives).sum()

    IoU = (TP + smooth) / (TP + FP + FN + smooth)

    DICE = 2 * IoU / (IoU + 1)

    return IoU, DICE
def calculate_iou_dice(pred_path, mask_path, smooth=1e-5):
    pre_img = Image.open(pred_path)
    pred = transform(pre_img)

    mask_img = Image.open(mask_path)
    mask = transform(mask_img)
    # 确保pred和target的大小一致
    assert pred.size() == mask.size(), "Size of predictions and targets must be the same"

    # 将pred和target转换为布尔型，即0和1，1代表前景，0代表背景
    pred_positives = pred.unsqueeze(0)
    mask_positives = mask.unsqueeze(0)

    return mean_iou(pred_positives, mask_positives)


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)
    dn = torch.where(dn > (ma - mi) / 2.0, 1.0, 0)
    return dn


def save_output(image_name, pred, d_dir):
    # pred = normPRED(pred)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('L')

    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    img_name = image_name.split(os.sep)[-1]

    aaa = img_name.split("/")
    imidx = aaa[-1]
    image_path = d_dir + '/' + imidx
    imo.save(image_path)
    return image_path