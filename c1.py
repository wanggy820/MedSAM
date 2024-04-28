import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry
# 将图像转换为SAM模型期望的格式
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from statistics import mean
from tqdm import tqdm
from torch.nn.functional import threshold, normalize
import os
join = os.path.join

bbox_coords = {}
ground_truth_masks = {}

data_root = "datasets/ISBI/"
filePath = data_root + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
f = open(filePath, encoding="utf-8")
data = pd.read_csv(f)


for img, seg in zip(data["img"], data["seg"]):
    im = cv2.imread(data_root + seg)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if len(contours) < 1:
        continue
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

    bbox_coords[img] = np.array([maxX, maxY, maxX + maxW, maxY + maxH])
    gt_grayscale = cv2.imread(data_root + seg, cv2.IMREAD_GRAYSCALE)
    ground_truth_masks[img] = (gt_grayscale == 0)





def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# name = 'ISBI2016_ISIC_Part1_Training_Data/ISIC_0000016.jpg'
# image_path = f'{data_root}{name}'
# image = cv2.imread(image_path)
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(image)
# show_box(bbox_coords[name], ax[0])
# show_mask(ground_truth_masks[name], ax[0])
# plt.show()

model_type = 'vit_b'
checkpoint = 'work_dir/MedSAM/medsam_vit_b.pth'
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()


print("size:{}".format(sam_model.image_encoder.img_size))
transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
    image = cv2.imread(f'{data_root}{k}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(200) #sam_model.image_encoder.img_size
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size

# 设置超参数
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.BCELoss()
keys = list(bbox_coords.keys())


num_epochs = 100
losses = []

print("begin train")
for epoch in range(num_epochs):
    print(f'EPOCH: {epoch}')
    epoch_losses = []
    # Just train on the first 20 examples
    for k in keys:

        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']

        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)

            prompt_box = bbox_coords[k]
            box = transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            box_torch = box_torch[None, :]

            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (
        1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

        loss = loss_fn(binary_mask, gt_binary_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    losses.append(epoch_losses)

    print(f'Mean loss: {mean(epoch_losses)}')
    checkpoint = {
        "model": sam_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    model_save_path = "work_dir/ISBI"
    torch.save(checkpoint, join(model_save_path, "medsam_model_ISBI_latest.pth"))


mean_losses = [mean(x) for x in losses]

plt.plot(list(range(len(mean_losses))), mean_losses)
plt.title('Mean epoch loss')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.show()

