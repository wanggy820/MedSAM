import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('./datasets/MICCAI2023/val/image/A-90.png')
image = cv2.resize(image, None, fx=0.5, fy=0.5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "./work_dir/MedSAM/medsam_vit_b.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
# 自动生成采样点对图像进行分割
mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())
print(masks[0])

plt.figure(figsize=(16, 16))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()