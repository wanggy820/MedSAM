# 导入了一些库
import warnings

import torchvision
from torch.nn import functional as F
from segment_anything_u2net.build_u2net_sam import build_sam
from utils.data_convert import mean_iou, compute_loss, build_dataloader_u2net, build_dataloader_box
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn, optim


warnings.filterwarnings(action='ignore')

# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='MICCAI', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help='')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=100, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models_box/', help='model path directory')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    return parser.parse_known_args()[0]


criterion = nn.BCELoss()
def main(opt):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(234)
    if device == f'cuda:0':
        torch.cuda.manual_seed_all(234)
    #  脚本使用预先构建的架构（sam_model_registry['vit_b']）定义了一个神经网络模型，并设置了优化器（AdamW）和学习率调度。
    print(device, 'is available')
    print("Loading model...")
    epoch_add = 0
    lr = opt.lr

    model_path = "./models_no_box/"
    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"
    if not os.path.exists(checkpoint):
        checkpoint = None
    # checkpoint = None
    sam = build_sam(checkpoint=checkpoint)
    sam = sam.to(device=device)

    optimizer = optim.AdamW(sam.mask_decoder.parameters(),
                            lr=lr, betas=beta, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

    # 脚本在各个检查点保存训练模型的状态字典，如果模型在验证集上取得最佳平均IOU，则单独保存最佳模型。
    if len(os.listdir(model_path)) == 0:
        save_path = os.path.join(model_path, f"{opt.dataset_name}_model_{opt.epochs}_{opt.batch_size}_0")
        os.makedirs(save_path)
    else:
        save_path = os.path.join(model_path,
                                 f"{opt.dataset_name}_model_{opt.epochs}_{opt.batch_size}_" + str(
                                     len(os.listdir(model_path))))
        os.makedirs(save_path)

    print('Training Start')
    best_loss = 999999999

    best_mIOU = 0

    tr_pl_loss_list = []
    tr_pl_mi_list = []

    dataloaders = build_dataloader_u2net(sam, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers)
    for epoch in range(opt.epochs):
        train_loss_list = []
        train_miou_list = []

        sam.train()
        iterations = tqdm(dataloaders['train'])

        # 循环进行模型的多轮训练
        for train_data in iterations:
            # 将训练数据移到指定设备，这里是GPU
            train_input = train_data['image'].to(device)

            train_target_mask = train_data['mask'].to(device, dtype=torch.float32)

            prompt_box = train_data["prompt_box"].to(device)
            prompt_masks = train_data["prompt_masks"].to(device)
            # auxiliary_ratio_masks = train_data["auxiliary_ratio_masks"].to(device)
            # 对优化器的梯度进行归零
            optimizer.zero_grad()

            with torch.no_grad():
                # 使用 sam 模型的 image_encoder 提取图像特征，并使用 prompt_encoder 提取稀疏和密集的嵌入。在本代码中进行提示输入，所以都是None.
                train_encode_feature = sam.image_encoder(train_input)
                train_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points=None, boxes=prompt_box,
                                                                                     masks=prompt_masks)

            #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
            train_mask, train_IOU = sam.mask_decoder(
                image_embeddings=train_encode_feature,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=train_sparse_embeddings,
                dense_prompt_embeddings=train_dense_embeddings,
                multimask_output=False)

            low_res_pred = torch.sigmoid(train_mask)

            low_res = low_res_pred * prompt_masks
            # low_res = low_res * torch.where(auxiliary_ratio_masks > 0, 1, 0)

            # c2 = low_res_pred.squeeze().cpu()
            # # c3 = torch.where(c2 > 0.5, 255.0, 0.0)
            # torchvision.utils.save_image(c2, "low_res_pred.png")
            #
            # c21 = prompt_masks.squeeze().cpu()
            # # c31 = torch.where(c21 > 0, 255.0, 0.0)
            # torchvision.utils.save_image(c21, "prompt_masks.png")
            #
            # c21 = low_res.squeeze().cpu()
            # # c31 = torch.where(c21 > 0, 255.0, 0.0)
            # torchvision.utils.save_image(c21, "low_res.png")


            # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
            train_true_iou = mean_iou(low_res, train_target_mask, eps=1e-6)
            train_miou_list = train_miou_list + train_true_iou.tolist()

            train_loss_one = compute_loss(low_res, train_target_mask, train_IOU, train_true_iou)
            train_loss_one.backward()

            optimizer.step()

            train_loss_list.append(train_loss_one.item())
            # 学习率调整
            if epoch_add == 0:
                if opt.global_step < opt.warmup_steps:
                    lr_scale = opt.global_step / opt.warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 8e-4 * lr_scale
                opt.global_step += 1

            pbar_desc = "Model train loss --- "
            pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
            pbar_desc += f", total mIOU: {np.mean(train_miou_list):.5f}"
            iterations.set_description(pbar_desc)

        train_loss = np.mean(train_loss_list)
        train_miou = np.mean(train_miou_list)

        torch.cuda.empty_cache()
        tr_pl_loss_list.append(train_loss)
        tr_pl_mi_list.append(train_miou)

        # sam.eval()
        scheduler.step()

        model_path = model_path + opt.dataset_name + '_sam.pth'
        sam = sam.to(device)

        if best_mIOU < train_miou:
            best_loss = train_loss
            best_mIOU = train_miou
            best_path = save_path + f'/{opt.dataset_name}_sam_best.pth'
            torch.save(sam.state_dict(), best_path)
            sam = sam.to(device)
            f = open(os.path.join(save_path, 'best.txt'), 'w')
            f.write(f"Experimental Day: {datetime.now()}")
            f.write("\n")
            f.write(f"mIoU: {str(best_mIOU)}")
            f.write("\n")
            f.write(f"epochs:{opt.epochs}")
            f.write("\n")
            f.write(f"batch_size:{opt.batch_size}")
            f.write("\n")
            f.write(f"learning_rate:{opt.lr}")
            f.write("\n")
            f.write(f"Data_set : {opt.dataset_name}")
            f.close()

        print("epoch : {:3d}, train loss : {:3.4f}, train mIOU : {:3.4f}, best loss : {:3.4f}, best mIOU : {:3.4f})"
              .format(epoch + 1 + epoch_add, train_loss, train_miou, best_loss, best_mIOU))

        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 5 == 0:
            model_path1 = save_path + "/" + opt.dataset_name + "_sam_" + str(epoch + 1 + epoch_add) + '_' + str(
                round(lr, 10)) + '.pth'
            torch.save(sam.state_dict(), model_path1)
            sam = sam.to(device)

    # (2, 2) 形式的图使用matplotlib可视化训练进展，生成用于训练和验证平均IOU、训练和验证损失的图表。
    plt_dict = {
        "Train_mIoU": tr_pl_mi_list,
        "Train_Loss": tr_pl_loss_list,
    }
    print(plt_dict)
    plt.figure(figsize=(15, 15))
    for i, (key, item) in enumerate(plt_dict.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(opt.epochs), item, label=f"{key}")
        plt.title(f"{key}", fontsize=20)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel(f'{key.split("_")[-1]}', fontsize=15)
        plt.grid(True)

    plt.savefig(save_path + f'/{opt.dataset_name}_sam_{opt.epochs}_{opt.batch_size}_{opt.lr}_result.png')

# {'Train_mIoU': [0.6623069808748885, 0.7618775984601478], 'Train_Loss': [0.5310597625645724, 0.3650271041826768]}
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)