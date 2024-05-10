import argparse
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os

from U2_Net.U2netSegDataset import U2netSegDataset
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET
from model import U2NETP
import logging
from utils.data_convert import getDatasets

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, masks_v):

	loss0 = bce_loss(d0,masks_v)
	loss1 = bce_loss(d1,masks_v)
	loss2 = bce_loss(d2,masks_v)
	loss3 = bce_loss(d3,masks_v)
	loss4 = bce_loss(d4,masks_v)
	loss5 = bce_loss(d5,masks_v)
	loss6 = bce_loss(d6,masks_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

def dice_iou_function(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return dice, iou

def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--model_name", type=str, default='u2net', help="datasets")
    parser.add_argument("--datasets", type=str, default='DRIVE', help="datasets")
    parser.add_argument("--root_dir", type=str, default='../datasets/', help="root_dir")
    parser.add_argument("--batch_size", type=int, default=10, help="root_dir")
    parser.add_argument("--epoch", type=int, default=500, help="root_dir")
    return parser

def main():
    opt = get_argparser().parse_args()

    datasets = opt.datasets
    print("datasets:{}".format(datasets))
    image_list, mask_list = getDatasets(datasets, opt.root_dir, "train")

    epoch_num = opt.epoch
    batch_size = opt.batch_size
    model_dir = os.path.join(os.getcwd(), 'saved_models', opt.model_name + os.sep)

    # salobj_dataset = SalObjDataset(
    #     image_list=image_list,
    #     mask_list=mask_list,
    #     transform=transforms.Compose([
    #         RescaleT(320),
    #         RandomCrop(288),
    #         ToTensorLab(flag=0)]))

    train_dataset = U2netSegDataset(image_list, mask_list, input_size=(320, 320))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    train_num = len(train_dataloader)
    # ------- 3. define model --------
    # define the net
    net = U2NET(3, 1)

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

    for epoch in range(0, epoch_num):
        net.train()
        running_loss = 0.0
        running_tar_loss = 0.0

        total_iou = 0
        for i, (images, masks) in enumerate(train_dataloader):
            images = images.type(torch.FloatTensor).to(device)
            masks = masks.type(torch.FloatTensor).to(device)

            # images_v, masks_v = Variable(images.to(device), requires_grad=False), Variable(masks.to(device),
            #                                                                                 requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(images)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, masks)
            u2net_dice, u2net_iou = dice_iou_function(d0.detach().cpu().numpy(), masks.detach().cpu().numpy())
            print("u2net_dice:{}, u2net_iou:{}".format(u2net_dice, u2net_iou))

            total_iou = total_iou + u2net_iou

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            if i % 10 == 0 and i > 0:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("[epoch: %3d/%3d, batch: %5d/%5d, iter: %d] train loss: %3f, tar: %3f， IOU:%3f, time:%s" % (
                    epoch + 1, epoch_num, (i + 1), train_num, (i + 1), running_loss / (i + 1),
                    running_tar_loss / (i + 1), total_iou / (i + 1), current_time))

        if best_loss == 0 or best_loss > running_tar_loss:
            best_loss = running_tar_loss
            torch.save(net.state_dict(), model_file)

        itr = len(train_dataloader)
        print("[epoch: %3d/%3d] train loss: %3f, tar: %3f， IOU:%3f" % (
            epoch + 1, epoch_num, running_loss / itr,
            running_tar_loss / itr, total_iou / itr))
        logging.info("[epoch: %3d/%3d] train loss: %3f, tar: %3f， IOU:%3f" % (
            epoch + 1, epoch_num, running_loss / itr,
            running_tar_loss / itr, total_iou/ itr))


if __name__ == '__main__':
    main()

