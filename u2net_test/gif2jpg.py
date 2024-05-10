import glob
import imageio
import numpy as np

from utils.data_convert import getDatasets
from PIL import Image

# image_list, mask_list = getDatasets("DRIVE", "../datasets/", "train")
# mask_full_path = ""
# gif_images = imageio.mimread(mask_full_path)

def gif2jpg(inupt_path, output_path):
    list = sorted(glob.glob(inupt_path))
    for path in list:
        gif_images = imageio.mimread(path)
        image = gif_images[0]
        image = Image.fromarray(image.astype(np.uint8))

        arr = path.split("/")
        image_name = arr[len(arr) - 1]
        arr = image_name.split(".")
        image_name = arr[0]
        # 保存为图片
        image.save(f'{output_path}/{image_name}.jpg')

inupt_path = "../datasets/DRIVE/test/1st_manual/*"
output_path = "../datasets/DRIVE/test/mask_jpg"
gif2jpg(inupt_path, output_path)