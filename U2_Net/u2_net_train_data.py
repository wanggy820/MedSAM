import time
import numpy as np
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
import argparse
from model import U2NET
from model import U2NETP
import logging
from utils.data_convert import getDatasets, mean_iou
from tqdm import tqdm

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Thyroid_tn3k', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    return parser.parse_known_args()[0]

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def main(opt):

    dataset_name = opt.dataset_name
    model_name = 'u2net'  # 'u2netp'
    # ------- 2. set the directory of training dataset --------
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    logging.basicConfig(filename=model_name + '_train_' + dataset_name + '.log', encoding='utf-8', level=logging.DEBUG)

    epoch_num = opt.epochs
    batch_size_train = opt.batch_size

    image_list, mask_list, auxiliary_list = getDatasets(dataset_name, opt.data_dir, "train")

    salobj_dataset = SalObjDataset(
        img_name_list=image_list,
        lbl_name_list=mask_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    train_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=False, num_workers=opt.num_workers)

    train_num = len(train_dataloader)
    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)

    model_file = model_dir + "u2net_bce_best_" + dataset_name + ".pth"
    if os.path.exists(model_file):
        net.load_state_dict(torch.load(model_file))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")

    best_loss = 0
    for epoch in range(0, epoch_num):
        net.train()

        train_loss_list = []
        train_miou_list = []
        best_loss = 999999999
        best_mIOU = 0
        iterations = tqdm(train_dataloader)
        for i, data in enumerate(iterations):
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
            iou, dice = mean_iou(d0, labels_v)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            train_miou_list = train_miou_list + iou.tolist()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            pbar_desc = "Model train loss --- "
            pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
            pbar_desc += f", total mIOU: {np.mean(train_miou_list):.5f}"
            iterations.set_description(pbar_desc)

        train_loss = np.mean(train_loss_list)
        train_miou = np.mean(train_miou_list)
        if best_mIOU < train_miou:
            best_loss = train_loss
            best_mIOU = train_miou
            torch.save(net.state_dict(), model_file)

        print("epoch : {:3d}, train loss : {:3.4f}, train mIOU : {:3.4f}, best loss : {:3.4f}, best mIOU : {:3.4f})"
              .format(epoch + 1, train_loss, train_miou, best_loss, best_mIOU))
        logging.info("epoch : {:3d}, train loss : {:3.4f}, train mIOU : {:3.4f}, best loss : {:3.4f}, best mIOU : {:3.4f})"
              .format(epoch + 1, train_loss, train_miou, best_loss, best_mIOU))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)