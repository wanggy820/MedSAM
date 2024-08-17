# -*- coding: utf-8 -*-
import os
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.autograd import Variable
import sys
import time

from PyQt5.QtGui import (
    QPainter,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
)

import numpy as np
from skimage import transform, io
import torch
from torch.nn import functional as F
from PIL import Image
from U2_Net.data_loader import RescaleT, ToTensorLab, SalObjDataset
from U2_Net.model import U2NET
from segment_anything import sam_model_registry
from utils.box import find_bboxes

# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
# MedSAM_CKPT_PATH = "models_no_box/MICCAI_sam_best.pth"
MEDSAM_IMG_INPUT_SIZE = 1024

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


print("Loading MedSAM model, a sec.")
tic = time.perf_counter()

# set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

print(f"Done, took {time.perf_counter() - tic}")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 255, 255),
]

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma-mi)/2.0, 1.0, 0)
    return dn

def find_u2net_bboxes(input, image_name):
    # normalization
    pred = input[:, 0, :, :]
    masks = normPRED(pred)

    predict = masks.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    # imo.save("33.png")
    return find_bboxes(imo)

def get_u2net_bbox(img_path):
    model_dir = "U2_Net/saved_models/u2net/u2net_bce_best_MICCAI.pth"
    img_name_list = [img_path]
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    prediction_dir = "./"
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        bboxes = find_u2net_bboxes(d1, img_name_list[i_test])
        return bboxes

class Window(QWidget):
    def __init__(self):
        super().__init__()

        # configs
        self.half_point_size = 5  # radius of bbox starting and ending points

        # app stats
        self.image_path = None
        self.color_idx = 0
        self.bg_img = None
        self.is_mouse_down = False
        self.rect = None
        self.point_size = self.half_point_size * 2
        self.start_point = None
        self.end_point = None
        self.start_pos = (None, None)
        self.embedding = None
        self.prev_mask = None

        self.view = QGraphicsView()
        self.view.setRenderHint(QPainter.Antialiasing)

        pixmap = self.load_image()

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.view)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")

        hbox = QHBoxLayout(self)
        hbox.addWidget(load_button)
        hbox.addWidget(save_button)

        vbox.addLayout(hbox)

        self.setLayout(vbox)

        # keyboard shortcuts
        self.quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.quit_shortcut.activated.connect(lambda: quit())

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

    def undo(self):
        if self.prev_mask is None:
            print("No previous mask record")
            return

        self.color_idx -= 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.prev_mask.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

        self.mask_c = self.prev_mask
        self.prev_mask = None

    def load_image(self):
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        self.img_3c = img_3c
        self.image_path = file_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        # self.scene.mousePressEvent = self.mouse_press
        # self.scene.mouseMoveEvent = self.mouse_move
        # self.scene.mouseReleaseEvent = self.mouse_release


        box_np = get_u2net_bbox(self.image_path)

        H, W, _ = self.img_3c.shape
        # print("bounding box:", box_np)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)

        if len(sam_mask.shape) > 2:
            sum_np = 0
            for i in range(0, sam_mask.shape[0]):
                sum_np += sam_mask[i]
        else:
            sum_np = sam_mask

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sum_np != 0] = colors[self.color_idx % len(colors)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        # self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))
        for i in range(0, box_np.shape[0]):
            self.scene.addRect(
                box_np[i][0], box_np[i][1], box_np[i][2] - box_np[i][0], box_np[i][3] - box_np[i][1], pen=QPen(QColor("red"))
            )

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.scene.removeItem(self.end_point)
        self.end_point = self.scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False

        if self.rect is not None:
            self.scene.removeItem(self.rect)

        bboxs = get_u2net_bbox(self.image_path)
        for j in bboxs:
            self.scene.addRect(
                j[0], j[1], j[2], j[3], pen=QPen(QColor("red"))
            )
            xmin = j[0]
            ymin = j[1]
            xmax = j[0] + j[2]
            ymax = j[1] + j[3]

        H, W, _ = self.img_3c.shape
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        # print("bounding box:", box_np)
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)

        self.prev_mask = self.mask_c.copy()
        self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
        self.color_idx += 1

        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        img = Image.blend(bg, mask, 0.2)

        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")
        img_1024 = transform.resize(
            self.img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # if self.embedding is None:
        with torch.no_grad():
            self.embedding = medsam_model.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)
        print("Done.")


app = QApplication(sys.argv)

w = Window()
w.show()

app.exec()


'''
# 3d标注工具使用  pair
#
# 标注成本，时间
#
# 经典 半监督，无监督的目标检测 论文， 标注时间 --> 摘要
#
# 论文中的数据集，训练后比对性能指标
#
# 加新的网络 u-net



 -------------医学3d交互式分割标注工具-------------

Pair 动态目标分割智能标注功能——intelligent Moving Object Segmentation (iMOS)
iMOS 教程:https://www.bilibili.com/video/BV1Gh4y1N7M2
iMOS适用于 CT/MRI/内窥镜/手术机器人/超声/造影等模态

ITK-SNAP 
http://www.itksnap.org/pmwiki/pmwiki.php

用于无缝3D导航的链接光标
一次在三个正交平面上手动分割
基于Qt的现代图形用户界面
支持许多不同的3D图像格式，包括NIfTI和DICOM
支持多个图像的并发，链接查看和分段
支持彩色，多通道和时变图像
3D切割平面工具，用于快速后处理分割结果
丰富的教程和视频文档

3D Slicer
三维体数据一般为DICOM格式或者NIFIT格式





 -------------医学数据集-------------

BRaTS 2021 Task 1 Dataset   13GB
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

https://bj.bcebos.com/ai-studio-online/c39f8954b2f740b3950cd3bef46062c8cec91292921f40a6853735a2ab67f0c2?authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2022-09-04T15%3A26%3A59Z%2F-1%2F%2Fb1e2c80998621dc75cbf0afed3749c16b8b2723eb4f6bb8a315f1ff6649adca2&responseContentDisposition=attachment%3B%20filename%3DBRATS2015.zip


MICCAI2020:https://www.miccai2020.org/en/  这个网站是MICCAI的官网
MICCAI比赛汇总：http://www.miccai.org/events/challenges/
BraTS2020：https://www.med.upenn.edu/cbica/brats2020/（BraTS目前有2015-2022年）

第二届青光眼竞赛：https://refuge.grand-challenge.org/
PET/CT三维头颈部肿瘤分割：https://www.aicrowd.com/challenges/hecktor
解剖脑部肿瘤扩散屏障分割：https://abcs.mgh.harvard.edu/
冠状动脉的自动分割：https://asoca.grand-challenge.org/
延时增强心脏MRI心肌缺血的自动评估：http://emidec.com/
脑动脉瘤检测和分析:https://cada.grand-challenge.org/Timeline/
计算精准医学放射学-病理学竞赛：脑肿瘤分类:https://miccai.westus2.cloudapp.azure.com/competitions/1
糖尿病足溃疡竞赛:https://dfu-challenge.github.io/
皮肤镜黑素瘤诊断:https://challenge.isic-archive.com/
颅内动脉瘤检测和分割竞赛:http://adam.isi.uu.nl/
大规模椎骨分割竞赛:https://verse2020.grand-challenge.org/
基于多序列CMR的心肌病理分割挑战:http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/MyoPS20/data1.html
肋骨骨折的检测和分类挑战:https://ribfrac.grand-challenge.org/
超声图像中甲状腺结节的分割与分类:https://tn-scui2020.grand-challenge.org/Dates/

IEEE ISBI 的竞赛合集：https://biomedicalimaging.org/2020/wp-content/uploads/static-html-to-wp/data/dff0d41695bbae509355435cd32ecf5d/challenges.html


Grand Challenges:https://grand-challenge.org/challenges/ 目前正在进行的有两个比赛一个是10月结束（这一个比赛中分别有大脑，肾，前列腺），另一个尚未宣布
Dream Challenge：http://dreamchallenges.org/ 这个比赛很多比赛都结束了，最近有一个新冠肺炎的比赛正在进行


2021竞赛：
IEEE ISBI竞赛合集：https://biomedicalimaging.org/2021/

医学影像数据集汇总（持续更新）150个
https://blog.csdn.net/m0_52987303/article/details/136659841?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-136659841-blog-129404242.235^v43^pc_blog_bottom_relevance_base6&spm=1001.2101.3001.4242.1&utm_relevant_index=3



MACCAI  LIDC-IDRI ISBI  BraTS

AMOS22数据集
https://zenodo.org/records/7155725#.Y0OOCOxBztM.

https://www.codabench.org/competitions/1847/

RadImageNet数据集
https://github.com/BMEII-AI/RadImageNet
'''



'''

真实 dice  批量输出

u2-net  超参一样， 与交互式效果对比  

多分类


牙齿、皮肤癌分割任务效果  最新的论文摘要、结果
Swin-UNetr  训练模型

多数据集合并训练

**  半监督 -->完全监督



'''



'''

训练结果 ， 平均值 


牙齿、皮肤癌分割任务效果  最新的论文摘要、结果， 发表的期刊，写成word文档
Swin-UNetr  训练模型



谷歌学术 搜索 引用论文





引用数据集相关顶会、顶刊的最优结果

Sam  u2net 结合的模型


思路路线




https://github.com/MedMNIST/MedMNIST  数据集

https://challenge.isic-archive.com/data/#2017  ISIC

https://www.fc.up.pt/addi/ph2%20database.html  皮肤镜



甲状腺论文分析
https://pdf.sciencedirectassets.com/271150/1-s2.0-S0010482523X0002X/1-s2.0-S0010482522010976/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBQaCXVzLWVhc3QtMSJGMEQCIE%2FBvsY6JKmjvq4YyU4O6%2BO9kvD40APZl7PJTYHRwp8mAiAWp6oIoyQlDOVsc5HQj0wS21Ou9f6q%2BlhOXSOblw6UMSq8BQi9%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMGlsbYP4g5IAVLH8JKpAFIh%2FiSw3kdx1CZqwEYz2D0X%2FIudPNd2JgI%2FS15m9arUW5d9HS4GRRP1b175vccFJiyHfJ3k2HQ%2BmcCb4%2FxtgBo9p2zMnp89n8lQHDlMTB5ivWTN3QTYIg7DCRfvblOE%2FoMRw2MFc3z9llL0uKrG%2FWsB03VJtx%2Bj4NCAsBCpOP9S3Tl389AUUZC4KjuhzJ%2FYVjYlOnAuvVyTahaOBtZT5xeDQz01NFA6d6RYC2fssCTTkBIvAtfZwaLByhZwIIJyER3c4anh9AkDG3UWmcGB046uVQsk4k9zOgNndiyWuXnRPythHnDFxKyOdHfKg7s8m7Qw7%2B47a2R2X3p8BXHUQe5AgVFzM1iaJ9A1vR6UtVpPnrT6ZNOa9zNrw9UKnW2G51zFt4J6%2BeDyvLZlXCEunw9Oow37gfufCx%2FZytev5nivMwVqpnBU%2BwLJcXllMP%2B8LHPfMTL86QGHi9%2BP6p8qHfTS%2BC18hy76CynqOgYaYZIyneDkNwjBE%2BOh1BQJPqMwuNxZKs8h6NwtcrWIhebvoHNIaqL4WH5bltq02Rn%2FN7x800vx0bOXKAKlrSMulJiynq2fO4T5weTpyWFk9oVi6k6uXeFQ7s3fW8PId6HYz8KDFOZpuxZWrLwFF%2BgDjS6Sh%2BmDO4YRjoDuQ1QLam8mDtZDGviDD4hotIPKhBDwC0GmZKG2%2FoZkLcS%2B7bu684FWXMQ9yLW8FpEXxzTlQehx1pfW%2BT3A7RW%2BKiH8m%2BFJrJICfI2BWTZzmGZtsDJtXsV5fYlt6gczzL7I4a6uVlPPORXYWThFHlBceAVNnb8KfxTGDAdZTSheE%2FYx94MPE2OEUTy4G%2BDP70Kc7KcUyda9G3OqzwMPOjrrI3XEGL0yunwgkw6NbqswY6sgE3TdhICmXpfts%2BP3AE9Yvv5HewqenldftI8PTlUAq1iB%2Bwo5i2nXyHNknZvPHV8sLN2nQHixGwSSXIcAoIBbq%2FObfGt53g82jZzmA5fAaVWe6ZOqHx57%2Bwm8nLIIXVi667Rb8Fi%2BNoNadzQZLP45m83Vx6C02VgsNf%2BxfGDMWIMz5ddWnyZMffTwRDV3lKnEKSkkCCPhjHH%2Br2E7lce6KRS2Vfg2KB1adfSS7eWsD2H39s&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240625T123908Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRZ7VIZU5%2F20240625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=18e7bc413b074c8c98fe3fea627715be4622b95cedee81c4e7747e0ff5fae34a&hash=160ddaa3e0d3b412bc9b3819a50a90ae8b2011e4ab0bcd67313de6eedd43fa53&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0010482522010976&tid=spdf-3886109e-3b90-4180-ad4a-a88de71656a7&sid=10d697f831b458469f7990274569184eabaegxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=150556575d555754015a&rr=899504c4fe9f6039&cc=sg
确认结果 --->公式


• IoU (Intersection Over Union) = TP/(FP + FN);  论文有问题、与代码不一样  代码：TP / (TP + FP + FN)
• DICE (dice coefficient) = 2*TP/(FP + FN + 2 * TP);


***** medsam 评价标准


mask 毛刺
查看论文用的事原论文的结果还是自己训练的结果？？

gpt
	1.	诚实报告你的发现：如果你在重复他人的实验时发现结果达不到他们声称的效果，你可以在论文中提到这一点。这种情况在学术界并不罕见，重要的是诚实和透明。
	2.	提供自己的结果：如果你的论文需要与他人的工作进行对比，那么你应该使用自己训练的结果。这样可以确保你的对比是基于你自己严格控制的实验条件下的结果，而不是依赖于他人的报告。
	3.	详细描述实验条件：在你的论文中，详细描述你是如何进行实验的，包括数据集、模型、超参数等。这样，其他研究人员可以理解你的工作，并且如果他们想重复你的实验，也有足够的信息。
	4.	讨论差异：如果你的结果与他人的结果有显著差异，尝试分析和讨论可能的原因。这可能包括数据集的差异、实验条件的不同、模型实现的细微差别等。
	5.	引用原始结果：虽然你可能没有得到与他人论文中相同的结果，但你仍然可以引用他们的结果作为背景信息。然后在你的论文中详细说明你的发现和结果。
	最重要的是保持学术诚信和透明，提供足够的信息和分析，以便其他研究人员能够理解和评估你的工作。

豆包
在这种情况下，您应该使用自己训练的结果进行对比。
如果您发现他人论文的结果存在可疑之处，不能将其直接用于对比。因为使用不可靠或未经证实的数据可能会影响您论文的科学性和可信度。
在您的论文中，可以客观地提及他人论文结果与您所观察到的不一致，但需要以严谨和负责的态度进行，避免无端指责或猜测。同时，重点阐述您自己的研究过程、方法和所得出的结果。

文心一言
在撰写论文时，确保研究的准确性和可信度是至关重要的。当你发现其他论文中的结果与描述不符，而你需要在你的研究中进行对比分析时，应该采取一种既科学又严谨的方法来处理这种情况。

1. 核实他人论文的结果
首先，确认你理解并正确引用了其他论文的结果。有时候，误解或误读可能是造成差异的原因。检查原文中的数据、图表和结论，确保你没有遗漏或错误地解释了任何信息。

2. 分析差异的原因
如果确认他人论文中的结果确实与描述不符，尝试分析这种差异的可能原因。这可能包括实验条件的不同、数据处理方法的差异、样本选择的偏差，或是论文中可能存在的错误等。

3. 决定是否使用他人论文的结果
如果差异显著且原因不明：谨慎考虑是否在你的论文中直接使用这些结果作为对比。可以提及这些差异，并指出由于这些不确定性，你选择了其他更可靠的数据源或方法进行比较。
如果差异可以合理解释：你可以在你的论文中讨论这些差异，并解释为什么你认为这些差异是合理的，然后基于你的理解使用这些数据作为对比。
4. 使用自己训练的结果
使用自己训练的结果进行对比通常是更可靠和有力的方法，因为它基于你自己的实验条件和数据处理流程。这可以确保对比的公平性和一致性，减少外部因素对结果的影响。

5. 透明度和公正性
无论你选择使用哪种方法，都应该在论文中清楚地说明你的选择理由，并保持透明度和公正性。如果使用了他人论文的结果，务必准确引用并注明可能的差异和不确定性。如果使用了自己训练的结果，也要详细说明实验条件、数据处理方法和结果验证过程。

结论
在对比研究中，确保使用可靠和一致的数据是至关重要的。如果可能，使用自己训练的结果进行对比是更理想的选择，因为它提供了更高的可控性和一致性。然而，在必要时，也可以谨慎地使用他人论文的结果，但务必注意透明度和公正性，以及合理解释任何潜在的差异和不确定性

更换loss重新训练

训练CPFNet




sam 
ma-sam  adapter
AutoSAM
DeSAM 解耦分割模型(Decoupling Segment Anything Model, DeSAM)
SurgicalSAM

UR-SAM
SemiSAM


多头注意力使用




USFM: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis

tn3k  ddti 谷歌学术

总结不同论文

gonghaifan@foxmail.com
'''


