import cv2
import pandas as pd

from utils.data_convert import getDatasets

data_type = "test"
image_list, mask_list, auxiliary_list = getDatasets("Thyroid_tn3k", root_dir="./datasets/", data_type=data_type)

width = []
height = []
image_name = []
label_name = []
bbox_x = []
bbox_y = []
bbox_width = []
bbox_height = []
for mask_path in mask_list:
    array = mask_path.split("\\")
    name = array[-1]
    mask = cv2.imread(mask_path)
    h, w, _ = mask.shape
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    _, mask1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 获取mask的边界框
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)

        width.append(w)
        height.append(h)
        image_name.append(name)
        label_name.append("thyroid")
        bbox_x.append(bx)
        bbox_y.append(by)
        bbox_width.append(bw)
        bbox_height.append(bh)



# label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,width,height
data = {
    "label_name" : label_name,
    "bbox_x" : bbox_x,
    "bbox_y" : bbox_y,
    "bbox_width" : bbox_width,
    "bbox_height" : bbox_height,
    "image_name" : image_name,
    "width" : width,
    "height" : height,
}

df = pd.DataFrame(data)

filename = f'./datasets/Thyroid_Dataset/tn3k/{data_type}_annotations.csv'
df.to_csv(filename, index=False)