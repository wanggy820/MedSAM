import numpy as np
import cv2


def dice_iou(pred, target, smooth=1e-5):
    # 读取并转换图像为二值化形式
    image1 = cv2.imread(pred, 0)
    c = image1.sum().item()
    image2 = cv2.imread(target, 0)
    _, image1_binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2_binary = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # 计算交集和并集
    intersection = cv2.bitwise_and(image1_binary, image2_binary)
    union = cv2.addWeighted(image1_binary, 0.5, image2_binary, 0.5, 0)

    # 计算DICE系数
    num_pixels_intersecting = cv2.countNonZero(intersection)
    num_pixels_total = cv2.countNonZero(union)
    dice_coefficient = (2 * num_pixels_intersecting+smooth) / float(num_pixels_total + num_pixels_intersecting+smooth)
    iou_coefficient = (num_pixels_intersecting + smooth) / float(num_pixels_total+smooth)
    return dice_coefficient, iou_coefficient

pre = '/Users/wang/Desktop/MedSAM-main/datasets/MICCAI2023/val/image/A-908_mask.png'
target = '/Users/wang/Desktop/MedSAM-main/datasets/MICCAI2023/val/mask/A-908.png'
dice_coefficient, iou_coefficient = dice_iou(pre, target)
print("DICE系数为:", dice_coefficient, "IOU:",iou_coefficient)