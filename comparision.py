import cv2
import numpy as np
from matplotlib import pyplot as plt

my_model_val_log = "./MySegmentModel/save_models/Thyroid_tn3k_fold0/vit_b_1.00/val.log"
sam_val_log = "./save_models/Thyroid_tn3k_fold0/vit_h_1.00/val.log"
bpat_unet_val_log = "./BPAT_UNet/val.log"


def get_image_path(str):
    lenght = len(str)
    name = str[(lenght - 11) : (lenght - 3)]
    print(name)
    return name

def get_image_dice(str):
    array = str.split("interaction dice:")
    dice = array[-1][0:8]
    return float(dice)

def set_plt(plt, index, path):
    plt.subplot(row, col, index)
    image = cv2.imread(path)
    plt.imshow(image, 'gray', extent=[0, 200, 0, 200])
    plt.xticks([]), plt.yticks([])
    # plt.axis('off')



class MyData():
    def __init__(self, name, bapt_dice, sam_dice, my_dice):
        super().__init__()
        self.name = name
        self.image_path = "./datasets/Thyroid_Dataset/tn3k/test-image/" + name
        self.mask_path = "./datasets/Thyroid_Dataset/tn3k/test-mask/" + name
        self.bapt_mask_path = "./BPAT_UNet/results/test-TN3K/BPAT-UNet/fold0/" + name
        self.sam_mask_path = "./save_models/Thyroid_tn3k_fold0/vit_h_1.00/val/" + name
        self.my_mask_path = "./MySegmentModel/save_models/Thyroid_tn3k_fold0/vit_b_1.00/val/" + name

        self.bapt_dice = bapt_dice
        self.sam_dice = sam_dice
        self.my_dice = my_dice
        self.interval = my_dice - bapt_dice

col = 5
row = 6

my_list = []
with open(bpat_unet_val_log, "r") as bapt_file:
    with open(sam_val_log, "r") as sam_file:
        with open(my_model_val_log, "r") as my_file:
            l = 0
            for bpat, sam, my in zip(bapt_file, sam_file, my_file):
                print(bpat)
                if l % 3 == 0:
                    name = get_image_path(bpat)
                    if name not in sam and name not in my:
                        assert 1, "error"
                elif l % 3 == 1:
                    bapt_dice = get_image_dice(bpat)
                    sam_dice = get_image_dice(sam)
                    my_dice = get_image_dice(my)
                    my_data = MyData(name, bapt_dice, sam_dice, my_dice)

                    i = 0
                    for index, data in enumerate(my_list):
                        if index > row:
                            break
                        if data.interval > my_data.interval:
                            i = index + 1
                            continue

                    if i < row:
                        my_list.insert(i, my_data)
                    if len(my_list) > row:
                        my_list.remove(my_list[-1])

                l = l + 1

fig = plt.figure(figsize=(6, 6))
i = 0


for index, data in enumerate(my_list):
    print(data.__dict__)

    set_plt(plt, i + 1, data.image_path)

    plt.ylabel(data.name, fontsize=10)
    set_plt(plt, i + 2, data.mask_path)

    set_plt(plt, i + 3, data.bapt_mask_path)
    set_plt(plt, i + 4, data.sam_mask_path)
    set_plt(plt, i + 5, data.my_mask_path)

    i += col

xticks_labels = ["origin", "mask", "BAPT", "sam+prompt", "ours"]
for index in range(len(xticks_labels)):
    plt.subplot(row, col, (row -1)*col + index + 1)
    plt.xlabel(xticks_labels[index], fontsize=10)

plt.tight_layout()
plt.show()




