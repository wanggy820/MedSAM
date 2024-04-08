import csv
import glob
import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import json


def transform_data(mask_path, catagrory_id):
    img1 = cv2.imread(mask_path)
    img1[img1 > 0] = catagrory_id
    image = Image.fromarray(np.uint8(img1))
    array = mask_path.split("/")
    name = array[len(array) - 1]
    transform_path = "test_pre_data"
    if (not os.path.exists(transform_path)):
        os.mkdir(transform_path)
    transform_path += "/catagrory_id_" + str(catagrory_id) + "_name_" + name
    image.save(transform_path)
    return "../" + transform_path

def pre_BUSI(datasets_dir, catagrory_id):
    tra_img_name_list = []
    tra_lbl_name_list = []
    # datasets_dir = "./datasets/Dataset_BUSI_with_GT/benign"
    file_list = os.listdir(datasets_dir)
    for f in file_list:
        if "_mask" not in f:
            array = f.split(".")
            if len(array) != 2:
                continue
            mask = array[0] + "_mask." + array[1]
            mask_path = datasets_dir + os.sep + mask
            if not os.path.exists(mask_path):
                continue
            file_path = "." + datasets_dir + os.sep + f
            transform_path = transform_data(mask_path, catagrory_id)

            tra_img_name_list.append(file_path)
            tra_lbl_name_list.append(transform_path)


    tra_img_name_list = sorted(tra_img_name_list)
    tra_lbl_name_list = sorted(tra_lbl_name_list)
    return tra_img_name_list, tra_lbl_name_list

def pre_ICBI(catagrory_id):
    datasets_dir = "./datasets/ICBI/"
    filePath = datasets_dir + "ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
    f = open(filePath, encoding="utf-8")
    data = pd.read_csv(f)
    tra_img_name_list = []
    tra_lbl_name_list = []

    for img, seg in zip(data["img"], data["seg"]):
        tra_img_name_list.append("." + datasets_dir + img)
        transform_path = transform_data(datasets_dir + seg, catagrory_id)
        tra_lbl_name_list.append(transform_path)
    return tra_img_name_list, tra_lbl_name_list

def pre_MICCAI(catagrory_id):
    img_name_list = glob.glob("./datasets/MICCAI2023/train/image/*")
    lbl_name_list = glob.glob("./datasets/MICCAI2023/train/mask/*")
    img_name_list = sorted(img_name_list)
    lbl_name_list = sorted(lbl_name_list)

    tra_img_name_list = []
    tra_lbl_name_list = []
    for img, mask_path in zip(img_name_list, lbl_name_list):
        tra_img_name_list.append("." + img)
        transform_path = transform_data(mask_path, catagrory_id)
        tra_lbl_name_list.append(transform_path)
    return tra_img_name_list, tra_lbl_name_list

def pre_Thyroid(catagrory_id):
    dir = "./datasets/Thyroid_Dataset/tg3k/"
    format = ".jpg"
    tra_img_name_list = []
    tra_lbl_name_list = []
    with open( dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for name in data["train"]:
            file_path = "." + dir + "Thyroid-image/" + "{:04d}".format(name) + format
            mask_path = dir + "Thyroid-mask/" + "{:04d}".format(name) + format
            transform_path = transform_data(mask_path, catagrory_id)
            tra_img_name_list.append(file_path)
            tra_lbl_name_list.append(transform_path)
    return tra_img_name_list, tra_lbl_name_list

def main():
    BUSI_dir = "./datasets/Dataset_BUSI_with_GT/"
    print(BUSI_dir + "benign")
    BUSI_image_list1, BUSI_mask_list1 = pre_BUSI(BUSI_dir + "benign", catagrory_id=1)
    print("BUSI image:", len(BUSI_image_list1), "BUSI mask :", len(BUSI_mask_list1))
    print(BUSI_dir + "malignant")
    BUSI_image_list2, BUSI_mask_list2 = pre_BUSI(BUSI_dir + "malignant", catagrory_id=2)
    print("BUSI image:", len(BUSI_image_list2), "BUSI mask :", len(BUSI_mask_list2))
    print(BUSI_dir + "normal")
    BUSI_image_list3, BUSI_mask_list3 = pre_BUSI(BUSI_dir + "normal", catagrory_id=3)
    print("BUSI image:", len(BUSI_image_list3), "BUSI mask :", len(BUSI_mask_list3))
    print("ISBI")
    ICBI_image_list, ICBI_mask_list = pre_ICBI(catagrory_id=4)
    print("ISBI image:", len(ICBI_image_list), "ISBI mask :", len(ICBI_mask_list))
    print("MICCAI")
    MICCAI_image_list, MICCAI_mask_list = pre_MICCAI(catagrory_id=5)
    print("MICCAI image:", len(MICCAI_image_list), "MICCAI mask :", len(MICCAI_mask_list))
    print("Thyroid")
    Thyroid_image_list, Thyroid_mask_list = pre_Thyroid(catagrory_id=6)
    print("Thyroid image:", len(Thyroid_image_list), "Thyroid mask :", len(Thyroid_mask_list))

    tra_img_name_list = [BUSI_image_list1, BUSI_image_list2, BUSI_image_list3, ICBI_image_list, MICCAI_image_list, Thyroid_image_list]
    tra_lbl_name_list = [BUSI_mask_list1, BUSI_mask_list2, BUSI_mask_list3, ICBI_mask_list, MICCAI_mask_list, Thyroid_mask_list]
    json_dict = {"image": tra_img_name_list, "mask": tra_lbl_name_list}

    json_data = json.dumps(json_dict, ensure_ascii=False)

    # 打开或创建一个 JSON 文件，以写入模式打开
    with open("pre_data.json", "w", encoding="utf-8") as file:
        # 将 JSON 数据写入文件
        file.write(json_data)


if __name__ == '__main__':
    main()






