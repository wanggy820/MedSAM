import json
import shutil

def pre_Thyroid():
    dir = "../datasets/Thyroid_Dataset/tg3k/"
    format = ".jpg"

    with open( dir + "tg3k-trainval.json", 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for name in data["val"]:
            file_path = dir + "Thyroid-image/" + "{:04d}".format(name) + format
            mask_path = dir + "Thyroid-mask/" + "{:04d}".format(name) + format
            shutil.copy(file_path, r'image')
            shutil.copy(mask_path, r'mask')

pre_Thyroid()