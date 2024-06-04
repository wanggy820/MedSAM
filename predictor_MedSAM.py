import argparse
import cv2
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from skimage import transform

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 1024
bbox_shift = 20


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 155 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--dataset_name", type=str, default='MICCAI', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=False, help='is use box')
    parser.add_argument("--image_path", type=str, default='./datasets/MICCAI2023/val/image/a-900.png',
                        help="image path")
    parser.add_argument("--mask_path", type=str, default='./datasets/MICCAI2023/val/mask/a-900.png', help="mask path")
    return parser


def box(image):
    H, W = image.shape[-2:]
    y_indices, x_indices = np.where(image > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    box_np = np.array([x_min, y_min, x_max, y_max])
    return box_np


def main():
    opt = get_argparser().parse_args()

    dataset_name = opt.dataset_name

    pre_dataset = "./pre_" + dataset_name + "/"
    if not os.path.exists(pre_dataset):
        os.mkdir(pre_dataset)

    # --------- 3. model define ---------
    model_path = "./models_box/"
    if opt.use_box == False:
        model_path = "./models_no_box/"

    checkpoint = f"./{model_path}{dataset_name}_sam_best.pth"
    # set up model
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    ############################ image
    image = cv2.imread(opt.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_1024 = transform.resize(
        image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    ############################
    predictor.set_image(image_1024)  # conpute the image embedding only once
    ############################ mask
    mask_np = cv2.imread(opt.mask_path, cv2.IMREAD_GRAYSCALE)  # 读取掩码数据
    box_origin = box(mask_np)

    mask_1024 = transform.resize(
        mask_np, (1024, 1024), order=0, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    ############################ prompt_box
    prompt_box = box(mask_1024)
    ############################ prompt_mask

    if opt.use_box:
        mask_256 = transform.resize(
            mask_1024, (256, 256), order=0, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        mask_256 = (mask_256 - mask_256.min()) / np.clip(
            mask_256.max() - mask_256.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 1)
        prompt_masks = np.expand_dims(mask_256, axis=0).astype(np.float32)
    else:
        prompt_masks = None
    sam_mask, scores, logit = predictor.predict(point_coords=None, point_labels=None, box=prompt_box,
                                                mask_input=prompt_masks,
                                                multimask_output=False)  # 1024x1024, bool

    sam_origin = transform.resize(sam_mask[0].astype(np.uint8), (mask_np.shape[-2], mask_np.shape[-1]), order=0,
                                  preserve_range=True, mode='constant', anti_aliasing=False)  # (256, 256)
    # segs = np.zeros_like(mask_np, dtype=np.uint8)
    # segs[sam_origin == True] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    show_mask(sam_origin, plt.gca(), random_color=True)
    show_box(box_origin, plt.gca())

    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
