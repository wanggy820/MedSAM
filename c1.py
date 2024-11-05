# import numpy
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# import cv2
# from segment_anything import SamPredictor, sam_model_registry
# # 将图像转换为SAM模型期望的格式
# from collections import defaultdict
# import torch
# from segment_anything.utils.transforms import ResizeLongestSide
# from statistics import mean
# from tqdm import tqdm
# from torch.nn.functional import threshold, normalize
# import os
# from PIL import Image
# from  skimage import filters
# join = os.path.join
# image = Image.open("./pres.png")
# gray_image = image.convert("L")
# a = numpy.array(gray_image)
# b = (a > 1).sum()
# gray_image_cv = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
# # edges = cv2.Canny(gray_image_cv, 100, 200)
# # smooth = filters.median(gray_image_cv)
# smooth = cv2.medianBlur(gray_image_cv, 5)
# cv2.imshow("smooth", smooth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import torch.cuda
#
# print(torch.__version__)
# print(torch.cuda.is_available())
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
#
# # 加载数据集
# x, y = load_iris(return_X_y=True)
#
# # 分割数据集
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, stratify=y, random_state=0
# )
#
# # 创建K近邻分类器
# knn_classifier = KNeighborsClassifier()
#
# # 定义超参数搜索范围
# param_grid = {'n_neighbors': [1, 3, 5, 7]}
#
# # 创建网格搜索对象
# grid_search = GridSearchCV(
#     estimator=knn_classifier,  # 要优化的模型
#     param_grid=param_grid,  # 要搜索的超参数及其候选值
#     cv=5,  # 使用5折交叉验证
#     verbose=0,  # 不输出运行过程信息
#     scoring='accuracy',  # 使用准确率作为评估指标
#     refit=True,  # 找到最佳参数后重新拟合模型
#     n_jobs=-1  # 使用所有可用的CPU核心进行计算
# )
#
# # 执行网格搜索
# grid_search.fit(x_train, y_train)
#
# # 打印最优参数和训练集上的最好得分
# print('最优参数组合:', grid_search.best_params_, '训练集最好得分:', grid_search.best_score_)
#
# # 在测试集上评估模型
# test_accuracy = grid_search.score(x_test, y_test)
# print('测试集准确率:', test_accuracy)
import json
import shutil

with open("result.json", "r", encoding='utf-8') as file:
    data = json.load(file)
    for dict in data:
        image_path = dict['image_path']
        mask_path = dict['mask_path']
        a = image_path.split('/')
        dist_path = './output/image/' + a[len(a) - 1]
        shutil.copy(image_path, dist_path)

        a = mask_path.split('/')
        dist_path = './output/mask/' + a[len(a) - 1]
        shutil.copy(mask_path, dist_path)
