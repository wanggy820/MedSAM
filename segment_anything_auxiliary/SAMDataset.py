# data loader
from __future__ import print_function, division
import torch
from skimage import io, transform, color
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms


# ==========================dataset load==========================
def Rescale(image, mask, output_size):
    img = transform.resize(image, (output_size, output_size), mode='constant')
    lbl = transform.resize(mask, (output_size, output_size), mode='constant', order=0,
                           preserve_range=True)
    return img, lbl


def RandomCrop(image, mask, output_size):
    if random.random() >= 0.5:
        image = image[::-1]
        mask = mask[::-1]

    h, w = image.shape[:2]
    new_h = new_w = output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image = image[top: top + new_h, left: left + new_w]
    mask = mask[top: top + new_h, left: left + new_w]

    return image, mask

def ToTensor(image, mask, flag=0, num_class=1):
    tmpLbl = np.zeros(mask.shape)

    if num_class == 1:
        if (np.max(mask) < 1e-6):
            mask = mask
        else:
            mask = mask / np.max(mask)

    # change the color space
    if flag == 2:  # with rgb and Lab colors
        tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
        tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
        if image.shape[2] == 1:
            tmpImgt[:, :, 0] = image[:, :, 0]
            tmpImgt[:, :, 1] = image[:, :, 0]
            tmpImgt[:, :, 2] = image[:, :, 0]
        else:
            tmpImgt = image
        tmpImgtl = color.rgb2lab(tmpImgt)

        # nomalize image to range [0,1]
        tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
        tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
        tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
        tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
        tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
        tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

        tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
        tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
        tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
        tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
        tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
        tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

    elif flag == 1:  # with Lab color
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = image[:, :, 0]
            tmpImg[:, :, 1] = image[:, :, 0]
            tmpImg[:, :, 2] = image[:, :, 0]
        else:
            tmpImg = image

        tmpImg = color.rgb2lab(tmpImg)

        tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
        tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
        tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

        tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
        tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
        tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

    else:  # with rgb color
        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

    tmpLbl[:, :, 0] = mask[:, :, 0]

    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpLbl = mask.transpose((2, 0, 1))

    tmpImg = np.ascontiguousarray(tmpImg)
    tmpLbl = np.ascontiguousarray(tmpLbl)
    return torch.from_numpy(tmpImg), torch.from_numpy(tmpLbl)

def ToTenser256(image_1024):
    image_1024 = image_1024.unsqueeze(1)
    image_256 = torch.nn.functional.interpolate(image_1024, scale_factor=0.25,
                                               mode='bilinear',
                                               align_corners=False)
    image_256 = image_256.squeeze(1)
    return image_256

class SAMDataset(Dataset):
    def __init__(self, image_name_list, mask_name_list):
        self.image_name_list = image_name_list
        self.mask_name_list = mask_name_list

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_path = self.image_name_list[idx]
        mask_path = self.mask_name_list[idx]

        image = io.imread(image_path)
        label_3 = io.imread(mask_path)

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        image_1200, mask_1200 = Rescale(image, label, 1200)
        image_1024, mask_1024 = RandomCrop(image_1200, mask_1200, 1024)
        image_1024, mask_1024 = ToTensor(image_1024, mask_1024)
        image_256 = ToTenser256(image_1024)
        mask_256 = ToTenser256(mask_1024)

        h, w = label_3.shape[-2:]
        size = np.array([w, h])
        data = {
            'image_1024': image_1024,
            'image_256': image_256,
            'mask': mask_256,
            "image_path": image_path,
            "mask_path": mask_path,
            "size": size
        }
        return data
