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


https://github.com/MedMNIST/MedMNIST  数据集

https://challenge.isic-archive.com/data/#2017  ISIC

https://www.fc.up.pt/addi/ph2%20database.html  皮肤镜



甲状腺论文分析
https://pdf.sciencedirectassets.com/271150/1-s2.0-S0010482523X0002X/1-s2.0-S0010482522010976/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBQaCXVzLWVhc3QtMSJGMEQCIE%2FBvsY6JKmjvq4YyU4O6%2BO9kvD40APZl7PJTYHRwp8mAiAWp6oIoyQlDOVsc5HQj0wS21Ou9f6q%2BlhOXSOblw6UMSq8BQi9%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMGlsbYP4g5IAVLH8JKpAFIh%2FiSw3kdx1CZqwEYz2D0X%2FIudPNd2JgI%2FS15m9arUW5d9HS4GRRP1b175vccFJiyHfJ3k2HQ%2BmcCb4%2FxtgBo9p2zMnp89n8lQHDlMTB5ivWTN3QTYIg7DCRfvblOE%2FoMRw2MFc3z9llL0uKrG%2FWsB03VJtx%2Bj4NCAsBCpOP9S3Tl389AUUZC4KjuhzJ%2FYVjYlOnAuvVyTahaOBtZT5xeDQz01NFA6d6RYC2fssCTTkBIvAtfZwaLByhZwIIJyER3c4anh9AkDG3UWmcGB046uVQsk4k9zOgNndiyWuXnRPythHnDFxKyOdHfKg7s8m7Qw7%2B47a2R2X3p8BXHUQe5AgVFzM1iaJ9A1vR6UtVpPnrT6ZNOa9zNrw9UKnW2G51zFt4J6%2BeDyvLZlXCEunw9Oow37gfufCx%2FZytev5nivMwVqpnBU%2BwLJcXllMP%2B8LHPfMTL86QGHi9%2BP6p8qHfTS%2BC18hy76CynqOgYaYZIyneDkNwjBE%2BOh1BQJPqMwuNxZKs8h6NwtcrWIhebvoHNIaqL4WH5bltq02Rn%2FN7x800vx0bOXKAKlrSMulJiynq2fO4T5weTpyWFk9oVi6k6uXeFQ7s3fW8PId6HYz8KDFOZpuxZWrLwFF%2BgDjS6Sh%2BmDO4YRjoDuQ1QLam8mDtZDGviDD4hotIPKhBDwC0GmZKG2%2FoZkLcS%2B7bu684FWXMQ9yLW8FpEXxzTlQehx1pfW%2BT3A7RW%2BKiH8m%2BFJrJICfI2BWTZzmGZtsDJtXsV5fYlt6gczzL7I4a6uVlPPORXYWThFHlBceAVNnb8KfxTGDAdZTSheE%2FYx94MPE2OEUTy4G%2BDP70Kc7KcUyda9G3OqzwMPOjrrI3XEGL0yunwgkw6NbqswY6sgE3TdhICmXpfts%2BP3AE9Yvv5HewqenldftI8PTlUAq1iB%2Bwo5i2nXyHNknZvPHV8sLN2nQHixGwSSXIcAoIBbq%2FObfGt53g82jZzmA5fAaVWe6ZOqHx57%2Bwm8nLIIXVi667Rb8Fi%2BNoNadzQZLP45m83Vx6C02VgsNf%2BxfGDMWIMz5ddWnyZMffTwRDV3lKnEKSkkCCPhjHH%2Br2E7lce6KRS2Vfg2KB1adfSS7eWsD2H39s&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240625T123908Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRZ7VIZU5%2F20240625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=18e7bc413b074c8c98fe3fea627715be4622b95cedee81c4e7747e0ff5fae34a&hash=160ddaa3e0d3b412bc9b3819a50a90ae8b2011e4ab0bcd67313de6eedd43fa53&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0010482522010976&tid=spdf-3886109e-3b90-4180-ad4a-a88de71656a7&sid=10d697f831b458469f7990274569184eabaegxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=150556575d555754015a&rr=899504c4fe9f6039&cc=sg
确认结果 --->公式

'''
'''
cnn  + transfomer     各自优势   


训练集 难度分类 10%

DINO  LLM  -->prompt

都分开做实验
 
 
 
 (Supervised Fine-Tuning,监督微调)
大模型的SFT方式主要包括以下几种：

1、全参数微调（Full Parameter Fine Tuning）：涉及对模型的所有权重进行调整，以使其完全适应特定领域或任务。这种方法适用于拥有大量与任务高度相关的高质量训练数据的情况。
2、部分参数微调（Sparse Fine Tuning / Selective Fine Tuning）：
3、LoRA（Low-Rank Adaptation）：通过向模型权重矩阵添加低秩矩阵来进行微调，既允许模型学习新的任务特定模式，又能够保留大部分预训练知识。
4、P-tuning v2：基于prompt tuning的方法，仅微调模型中与prompt相关的部分参数，而不是直接修改模型主体的权重。
5、QLoRA：可能是指Quantized Low-Rank Adaptation或其他类似技术，它可能结合了低秩调整与量化技术，以实现高效且资源友好的微调。
6、冻结（Freeze）监督微调：在这种微调方式中，部分或全部预训练模型的权重被冻结，仅对模型的部分层或新增的附加组件进行训练。这样可以防止预训练知识被过度覆盖，同时允许模型学习针对新任务的特定决策边界。


RLHF Reinforcement Learning fromHuman Feedback，人类反馈强化学习）

dino 模型，  模型参数微调，  RL强化学习


encoder + prompt (冻结)

预测分支  ---iou_pred， 难样本

难样本----引入元学习  ------


Masked Image Training for Generalizable Deep Image Denoising
在训练过程中对输入图像的随机像素进行掩蔽，并在训练过程中重建缺失的信息。还在自注意力层中掩蔽特征，以避免训练和测试不一致性的影响

没有真实image！！！


1、iou  参与训练
2、寻找更多的prompt
3、 adapter
4、看一些相关的论文
5、feature_map




难样本提取

聚类 算法分离不同的结节

得到 难样本  ---参数
数据扩充---- 避免过拟合




论文： Masked-attention Mask Transformer for Universal Image Segmentation
Mask2Former的核心创新在于其遮蔽注意力机制。
通过限制交叉注意力的范围，使得模型能够专注于预测掩膜区域内的局部特征。
不仅提高了模型的收敛速度，而且在多个流行的数据集上取得了显著的性能提升。
是基于一个元架构，包含背景特征提取器、像素解码器和Transformer解码器。
这种设计使得Mask2Former不仅在性能上超越了现有的专用架构，而且在训练效率上也有明显的优势。通过引入多尺度高分辨率特征和一系列优化改进，Mask2Former在不增加计算量的情况下，实现了性能的显著提升。
此外，通过在随机采样点上计算掩膜损失，Mask2Former还大幅降低了训练过程中的内存消耗。

局限性：在处理小对象时的性能仍有提升空间，且在泛化到新任务时仍需要针对性的训练。




microsoft  generative-ai
https://github.com/microsoft/generative-ai-for-beginners/blob/main/02-exploring-and-comparing-different-llms/README.md

论文 BEIT， BEIT V2

https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/beit


https://blog.csdn.net/RichardsZ_/article/details/125708964
https://blog.csdn.net/oYeZhou/article/details/113770019
https://mp.weixin.qq.com/s?__biz=MzIwNDY0MjYzOA==&mid=2247514749&idx=1&sn=bfa35fd34561201469e9fcafd6a45a4f&chksm=968f621c76c6bbfcf94570a18f5f14511e6c8f50d39a20729660d07427a218675d6dd118028e&scene=27

MAE:Masked Autoencoder


ios 
https://www.jianshu.com/p/410f01d9e638
https://zhuanlan.zhihu.com/p/275986408





生成式 生成新数据集、


mask2former



英文会议（模版，周期短，过审率高） dino  初稿,  对比方法 （2024年，  dice低）

'''


