import glob
import time

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
import logging

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


def iou_function(a, b, epsilon=1e-5):
    # 首先将a和b按照0/1的方式量化
    a = (a.detach().cpu().numpy() > 0).astype(int)
    b = (b.detach().cpu().numpy() > 0).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou


def main():
    datasets = "MICCAI"
    model_name = 'u2net'  # 'u2netp'
    # ------- 2. set the directory of training dataset --------
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    logging.basicConfig(filename=model_name + '_train_' + datasets + '.log', encoding='utf-8', level=logging.DEBUG)

    epoch_num = 1000
    batch_size_train = 10

    tra_img_name_list = glob.glob("../datasets/MICCAI2023/train/image/*")
    tra_lbl_name_list = glob.glob("../datasets/MICCAI2023/train/mask/*")

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    train_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=False, num_workers=1)

    train_num = len(train_dataloader)
    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)

    model_file = model_dir + "u2net_bce_best_" + datasets + ".pth"
    if os.path.exists(model_file):
        net.load_state_dict(torch.load(model_file))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")


    best_loss = 0
    itr = 0
    for epoch in range(42, epoch_num):
        net.train()
        running_loss = 0.0
        running_tar_loss = 0.0

        total_iou = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                            requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            iou = iou_function(d0, labels_v)
            total_iou = (total_iou * i + iou)/(i+1)
            itr += 1
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            if i % 10 == 0 and i > 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f， IOU:%3f, time:%s" % (
                    epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, (i + 1), running_loss / (i + 1),
                    running_tar_loss / (i + 1), total_iou, current_time))

        if best_loss == 0 or best_loss > running_tar_loss:
            best_loss = running_tar_loss
            torch.save(net.state_dict(), model_file)

        print("[epoch: %3d/%3d] train loss: %3f, tar: %3f， IOU:%3f" % (
            epoch + 1, epoch_num, running_loss / itr,
            running_tar_loss / itr, total_iou))
        logging.info("[epoch: %3d/%3d] train loss: %3f, tar: %3f， IOU:%3f" % (
            epoch + 1, epoch_num, running_loss / itr,
            running_tar_loss / itr, total_iou))
        itr = 0

if __name__ == '__main__':
    main()

