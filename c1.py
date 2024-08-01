import numpy
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
from PIL import Image
from  skimage import filters
join = os.path.join
image = Image.open("/Users/wang/Downloads/0002.jpg")
gray_image = image.convert("L")
gray_image_cv = cv2.cvtColor(numpy.array(gray_image), cv2.COLOR_RGB2BGR)
# edges = cv2.Canny(gray_image_cv, 100, 200)
# smooth = filters.median(gray_image_cv)
smooth = cv2.medianBlur(gray_image_cv, 5)
cv2.imshow("smooth", smooth)
cv2.waitKey(0)
cv2.destroyAllWindows()




