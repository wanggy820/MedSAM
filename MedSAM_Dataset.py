import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image

def read_picture(file_path):
    with Image.open(file_path) as img:
        return np.array(img)
class MedSAM_Dataset(Dataset):
    def __init__(self, sam, image_list, mask_list):
        self.device = sam.device
        self.image_list = image_list
        self.mask_list = mask_list

        self.transform = ResizeLongestSide(1024)
        self.preprocess = sam.preprocess
        self.img_size = sam.image_encoder.img_size
        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        #####################################

        image_path = self.image_list[idx] # 读取image data路径
        mask_path = self.mask_list[idx] # 读取mask data 路径
        #####################################

        if mask_path.endswith(".gif"):
            img = read_picture(image_path)
            mask = read_picture(mask_path)
        else:
            img = cv2.imread(image_path)  # 读取原图数据
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform.apply_image(img) #
        img = torch.as_tensor(img) # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0) # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img.to(device=self.device)) # img nomalize or padding
        #####################################


        mask = self.transform.apply_image(mask) # 变换(1024)

        mask = torch.as_tensor(mask) # torch tensor
        mask = mask.unsqueeze(0)

        h, w = mask.shape[-2:]

        padh = self.img_size - h
        padw = self.img_size - w

        mask = F.pad(mask, (0, padw, 0, padh))
        mask = self.resize(mask).squeeze(0)
        mask = (mask != 0) * 1

        #####################################
        data = {
            'image': img,
            'mask': mask,
        }
        return data

