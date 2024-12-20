import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from BPAT_UNet.visualization.metrics import evaluate
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
from torchvision import transforms
from BPAT_UNet.dataloaders import custom_transforms as trforms, utils


class MedSAMBox(Dataset):
    def __init__(self, sam, auxiliary_model, image_list, mask_list, auxiliary_list, bbox_shift=0, ratio=1.02, data_type='train'):
        self.device = sam.device
        self.preprocess = sam.preprocess

        self.auxiliary_model = auxiliary_model

        self.image_list = image_list
        self.mask_list = mask_list
        self.auxiliary_list = auxiliary_list

        self.img_size = sam.image_encoder.img_size
        self.transform_image = ResizeLongestSide(self.img_size)

        self.output_size = 256
        self.transform_mask = ResizeLongestSide(self.output_size)

        self.bbox_shift = bbox_shift

        ratio = min(ratio, 2)
        ratio = max(ratio, 1)
        self.ratio = ratio
        self.data_type = data_type

    def __len__(self):
        return len(self.image_list)

    def preprocessMask(self, mask_np, transform, size):
        mask = transform.apply_image(mask_np)  #
        mask = torch.as_tensor(mask / 255.0)
        h, w = mask.shape[-2:]
        padh = size - h
        padw = size - w
        mask = F.pad(mask, (0, padw, 0, padh))
        return mask

    def __getitem__(self, idx):
        image_path = self.image_list[idx]  # 读取image data路径
        mask_path = self.mask_list[idx]  # 读取mask data 路径
        # auxiliary_path = self.auxiliary_list[idx]
        #####################################

        img = cv2.imread(image_path)  # 读取原图数据
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
        auxiliary_np = mask_np
        if self.data_type == 'train':
            r = random.random()  # 随机翻转
            if r < 0.25:
                img = np.flip(img)
                mask_np = np.flip(mask_np)
                auxiliary_np = np.flip(auxiliary_np)
            elif r < 0.5:
                img = np.rot90(img, k=1)
                mask_np = np.rot90(mask_np, k=1)
                auxiliary_np = np.rot90(auxiliary_np, k=1)
            elif r < 0.75:
                img = np.rot90(img, k=3)
                mask_np = np.rot90(mask_np, k=3)
                auxiliary_np = np.rot90(auxiliary_np, k=3)

            r = random.random()  # 随机裁剪
            H, W = img.shape[0], img.shape[1]
            if r < 0.8:
                top = np.random.randint(0, H / 8)
                left = np.random.randint(0, W / 8)
                img = img[top: H, left: W]
                mask_np = mask_np[top: H, left: W]
                auxiliary_np = auxiliary_np[top: H, left: W]
        else:
           auxiliary_np = self.get_auxiliary_np(image_path, mask_path, img, mask_np)

        img = self.transform_image.apply_image(img)  #
        img = torch.as_tensor(img)  # torch tensor 变更
        img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

        img = self.preprocess(img.to(device=self.device))  # img nomalize or padding
        # min = img.min()
        # max = img.max()
        # img = img + 0.1 * torch.randn(img.shape).to(self.device)
        # img = torch.clamp(img, min, max)

        image_256 = F.interpolate(img.unsqueeze(0), size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        image_256 = image_256.squeeze(0)
        #####################################

        mask_256 = self.preprocessMask(mask_np, self.transform_mask, self.output_size)
        mask_256 = torch.as_tensor(mask_256).unsqueeze(0)

        ##################################### 不能用 find_bboxes() 张量维度不一样

        if (auxiliary_np > 0).sum() < 200:
            auxiliary_np = (np.ones(auxiliary_np.shape) * 255).astype(np.float32)

        auxiliary_256 = self.preprocessMask(auxiliary_np, self.transform_mask, self.output_size)
        auxiliary_1024 = self.preprocessMask(auxiliary_np, self.transform_image, self.img_size)

        pt, point = self.fixed_click(np.array(auxiliary_1024))

        y_indices, x_indices = np.where(auxiliary_1024 > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            x_min = y_min = 0
            x_max = y_max = self.img_size
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(self.img_size, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(self.img_size, y_max + random.randint(0, self.bbox_shift))

        box_1024 = np.array([[x_min, y_min, x_max, y_max]])
        box_1024 = box_1024.astype(np.int16)

        #####################################
        size = int(self.output_size * self.ratio)
        transform_ratio = ResizeLongestSide(size)
        auxiliary_ratio = self.preprocessMask(auxiliary_np, transform_ratio, size)

        top = int(size * (self.ratio - 1) / 2)
        bottom = top + self.output_size
        left = int(size * (self.ratio - 1) / 2)
        right = left + self.output_size
        auxiliary_ratio_masks = auxiliary_ratio[top:bottom, left:right]

        auxiliary_ratio_masks += auxiliary_256
        auxiliary_ratio_masks = auxiliary_ratio_masks // 2 + auxiliary_ratio_masks % 2
        prompt_masks1 = torch.where(auxiliary_ratio_masks > 0, 255.0, 0.0)
        np_p = prompt_masks1.numpy()
        gray_image_cv = cv2.cvtColor(np_p, cv2.COLOR_RGB2BGR)
        smooth = cv2.medianBlur(gray_image_cv, 5)

        imo = Image.fromarray(np.uint8(smooth)).convert('L')
        auxiliary_ratio_masks1 = np.array(imo)
        prompt_masks = torch.from_numpy(auxiliary_ratio_masks1).unsqueeze(0).to(torch.float32)
        # prompt_masks = (auxiliary_ratio_masks * 255).to(torch.float32).unsqueeze(0)
        #####################################
        h, w = mask_np.shape[-2:]
        size = np.array([w, h])
        data = {
            'image': img,
            'image_256': image_256,
            'mask': mask_256,
            "prompt_box": box_1024,
            "prompt_masks": prompt_masks,
            "image_path": image_path,
            "mask_path": mask_path,
            "size": size,
            "pt": pt,
            "point": np.array(point)
        }
        return data

    def fixed_click(self, mask, class_id=1):
        indices = np.argwhere(mask == class_id)
        indices[:, [0, 1]] = indices[:, [1, 0]]
        point_label = 1
        if len(indices) == 0:
            point_label = 0
            indices = np.argwhere(mask != class_id)
            indices[:, [0, 1]] = indices[:, [1, 0]]
        pt = indices[len(indices) // 2]
        return pt[np.newaxis, :], [point_label]

    def get_auxiliary_np(self, image_path, mask_path, image_origin, mask_np):
        class_name = type(self.auxiliary_model).__name__
        with torch.no_grad():
            h, w = mask_np.shape[-2:]
            composed_transforms_ts = transforms.Compose([
                trforms.FixedResize(size=(256, 256)),
                trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                trforms.ToTensor()])
            image = Image.open(image_path).convert('RGB')
            label = np.array(Image.open(mask_path).convert('L'))
            label = label / label.max()
            label = Image.fromarray(label.astype(np.uint8))
            sample = {'image': image, 'label': label}
            sample = composed_transforms_ts(sample)
            image_a = sample['image'].unsqueeze(0)
            if class_name == 'MySAMModel':
                auxiliary_256, bian = self.auxiliary_model.auxiliary_model(image_a.to(self.device, dtype=torch.float32))
            else:
                auxiliary_256, bian = self.auxiliary_model(image_a.to(self.device, dtype=torch.float32))
            prob_pred = torch.sigmoid(auxiliary_256)
            prob_pred = F.interpolate(prob_pred, size=(h, w), mode='bilinear', align_corners=True)
            auxiliary_np = prob_pred.squeeze().detach().cpu().numpy() * 255
            auxiliary_np = auxiliary_np.astype(np.uint8)


            if class_name == 'MySAMModel':
                img = self.transform_image.apply_image(image_origin)  #
                img = torch.as_tensor(img)  # torch tensor 变更
                img = img.permute(2, 0, 1).contiguous()[None, :, :, :].squeeze(
                    0)  # (高, 宽, 通道) -> (通道, 高, 宽) 变更后 设置添加None

                img = self.preprocess(img.to(device=self.device))  # img nomalize or padding
                image_256 = F.interpolate(img.unsqueeze(0), size=(self.output_size, self.output_size), mode='bilinear',
                                          align_corners=False)
                image_256 = image_256.squeeze(0)
                #####################################

                mask_256 = self.preprocessMask(mask_np, self.transform_mask, self.output_size)
                mask_256 = torch.as_tensor(mask_256).unsqueeze(0)

                ##################################### 不能用 find_bboxes() 张量维度不一样

                if (auxiliary_np > 0).sum() < 200:
                    auxiliary_np = (np.ones(auxiliary_np.shape) * 255).astype(np.float32)

                auxiliary_256 = self.preprocessMask(auxiliary_np, self.transform_mask, self.output_size)
                auxiliary_1024 = self.preprocessMask(auxiliary_np, self.transform_image, self.img_size)

                pt, point = self.fixed_click(np.array(auxiliary_1024))

                y_indices, x_indices = np.where(auxiliary_1024 > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    x_min = y_min = 0
                    x_max = y_max = self.img_size
                else:
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    x_min = max(0, x_min - random.randint(0, self.bbox_shift))
                    x_max = min(self.img_size, x_max + random.randint(0, self.bbox_shift))
                    y_min = max(0, y_min - random.randint(0, self.bbox_shift))
                    y_max = min(self.img_size, y_max + random.randint(0, self.bbox_shift))

                box_1024 = np.array([[x_min, y_min, x_max, y_max]])
                box_1024 = box_1024.astype(np.int16)

                #####################################
                size = int(self.output_size * self.ratio)
                transform_ratio = ResizeLongestSide(size)
                auxiliary_ratio = self.preprocessMask(auxiliary_np, transform_ratio, size)

                top = int(size * (self.ratio - 1) / 2)
                bottom = top + self.output_size
                left = int(size * (self.ratio - 1) / 2)
                right = left + self.output_size
                auxiliary_ratio_masks = auxiliary_ratio[top:bottom, left:right]

                auxiliary_ratio_masks += auxiliary_256
                auxiliary_ratio_masks = auxiliary_ratio_masks // 2 + auxiliary_ratio_masks % 2
                prompt_masks1 = torch.where(auxiliary_ratio_masks > 0, 255.0, 0.0)
                np_p = prompt_masks1.numpy()
                gray_image_cv = cv2.cvtColor(np_p, cv2.COLOR_RGB2BGR)
                smooth = cv2.medianBlur(gray_image_cv, 5)

                imo = Image.fromarray(np.uint8(smooth)).convert('L')
                auxiliary_ratio_masks1 = np.array(imo)
                prompt_masks = torch.from_numpy(auxiliary_ratio_masks1).unsqueeze(0).to(torch.float32)
                # prompt_masks = (auxiliary_ratio_masks * 255).to(torch.float32).unsqueeze(0)
                #####################################
                h, w = mask_np.shape[-2:]
                size = np.array([w, h])
                data = {
                    'image': img.unsqueeze(0),
                    'image_256': image_256.unsqueeze(0),
                    'mask': mask_256.unsqueeze(0),
                    "prompt_box": torch.tensor(box_1024),
                    "prompt_masks": prompt_masks.unsqueeze(0),
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "size": size,
                    "pt": torch.tensor(pt),
                    "point": torch.tensor(point)
                }
                auxiliary_256, iou = self.auxiliary_model(data)

                auxiliary_np = auxiliary_256.squeeze().detach().cpu().numpy() * 255
                auxiliary_np = auxiliary_np.astype(np.uint8)

        return auxiliary_np