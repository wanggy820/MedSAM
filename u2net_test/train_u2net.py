# coding: utf-8
# author： hxy
# 20220420
"""
训练代码：u2net、u2netp
train it from scratch.
"""
import os
import datetime
import torch
import numpy as np
from tqdm import tqdm

from U2_Net.U2netSegDataset import U2netSegDataset
from U2_Net.model.u2net import U2NET, U2NETP
from torch.utils.data import DataLoader

from utils.data_convert import getDatasets

# 参考u2net源码loss的设定
bce_loss = torch.nn.BCELoss(reduction='mean')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    # loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(),
    # loss6.data.item()))

    return loss0, loss


def load_data(datasets, root_dir, batch_size, num_workers, input_size):
    """
    :param img_folder: 图片保存的fodler
    :param mask_folder: mask保存的fodler
    :param batch_size: batch_size的设定
    :param num_workers: 数据加载cpu核心数
    :param input_size: 模型输入尺寸
    :return:
    """
    image_list, mask_list = getDatasets(datasets, root_dir, "train")

    train_dataset = U2netSegDataset(image_list,
                                    mask_list,
                                    input_size=input_size)
    image_list, mask_list = getDatasets(datasets, root_dir, "test")
    val_dataset = U2netSegDataset(image_list,
                                  mask_list,
                                  input_size=input_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def dice_iou_function(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return dice, iou

def train_model(epoch_nums, cuda_device, model_save_dir):
    """
    :param epoch_nums: 训练总的epoch
    :param cuda_device: 指定gpu训练
    :param model_save_dir: 模型保存folder
    :return:
    """
    current_time = datetime.datetime.now()
    current_time = datetime.datetime.strftime(current_time, '%Y-%m-%d-%H:%M')
    model_save_dir = os.path.join(model_save_dir, current_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    else:
        pass


    device = torch.device(cuda_device)
    train_loader, val_loader = load_data("DRIVE",
                                         '../datasets/',
                                         batch_size=10,
                                         num_workers=1,
                                         input_size=(320, 320))

    # input 3-channels, output 1-channels
    net = U2NET(3, 1)
    # model_dir = "/Users/wang/Desktop/MedSAM-main/U2_Net/saved_models/u2net_human_seg/u2net_human_seg.pth"
    # net.load_state_dict(torch.load(model_dir))
    # net = U2NETP(3, 1)

    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net, device_ids=[6, 7])
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for epoch in range(0, epoch_nums):
        run_loss = list()
        run_tar_loss = list()

        net.train()
        for i, (inputs, gt_masks) in enumerate(tqdm(val_loader)):
            optimizer.zero_grad()
            inputs = inputs.type(torch.FloatTensor)
            gt_masks = gt_masks.type(torch.FloatTensor)
            inputs, gt_masks = inputs.to(device), gt_masks.to(device)

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, gt_masks)
            u2net_dice, u2net_iou = dice_iou_function(d0.detach().cpu().numpy(), gt_masks.detach().cpu().numpy())
            print("u2net_dice:{}, u2net_iou:{}".format(u2net_dice, u2net_iou))
            loss.backward()
            optimizer.step()

            run_loss.append(loss.item())
            run_tar_loss.append(loss2.item())
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("--Train Epoch:{}--".format(epoch))
        print("--Train run_loss:{:.4f}--".format(np.mean(run_loss)))
        print("--Train run_tar_loss:{:.4f}--\n".format(np.mean(run_tar_loss)))

        if epoch % 20 == 0:
            checkpoint_name = 'checkpoint_' + str(epoch) + '_' + str(np.mean(run_loss)) + '.pth'
            torch.save(net.state_dict(), os.path.join(model_save_dir, checkpoint_name))
            print("--model saved:{}--".format(checkpoint_name))


if __name__ == '__main__':
    train_model(epoch_nums=500, cuda_device='mps',
                model_save_dir='backup')

