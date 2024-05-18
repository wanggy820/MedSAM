import os
import cv2
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from U2_Net.data_loader import RescaleT
from U2_Net.data_loader import ToTensor
from U2_Net.data_loader import ToTensorLab
from U2_Net.data_loader import SalObjDataset

from U2_Net.model import U2NET  # full size version 173.6 MB
from U2_Net.model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp

    num_class = 7

    image_dir = os.path.join(os.getcwd(), 'val', 'image')
    model_dir = os.path.join(os.getcwd(), 'U2_Net', 'saved_models', model_name,
                             'u2net_bce_best_ALL.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, num_class)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, num_class)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']

        image = cv2.imread(img_name_list[i_test])
        image_name = os.path.basename(img_name_list[i_test])

        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        d1 = d1.squeeze(dim=0)  # torch.Size([1, 3, 320, 320]) -> torch.Size([3, 320, 320])

        d1 = F.softmax(d1, dim=0)  # [3, 320, 320]


        predict_np = torch.argmax(d1, dim=0, keepdim=True)
        # print(predict_np.shape)  # [1, 320, 320],3个类别，对应3个通道，获取概率值最高的下标
        predict_np = normPRED(predict_np)
        predict_np = predict_np.cpu().detach().numpy().squeeze()  # 转到cpu设备

        predict_np = cv2.resize(predict_np, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)  # resize和原图一样的大小



        r = predict_np.copy()
        b = predict_np.copy()
        g = predict_np.copy()


        cls = dict([(1, (0, 0, 255)), #蓝
                    (2, (255, 0, 255)), #紫
                    (3, (0, 255, 0)),  # 绿
                    (4, (255, 0, 0)),    #红色
                    (5, (255, 255, 0)),  #黄色
                    (6, (200, 255, 255))]) #纯蓝
        for c in cls:
            r[r == c] = cls[c][0]
            g[g == c] = cls[c][1]
            b[b == c] = cls[c][2]


        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        # print('类别', np.unique(predict_np))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        im = Image.fromarray(rgb.astype(np.uint8))
        im.save('./val/test/' + str(image_name)[:-4] + '.png')

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    # main()

    a = torch.ones([8, 4, 5, 6])
    b = torch.ones([1, 1, 5, 6])
    c = a + b
    print(c)