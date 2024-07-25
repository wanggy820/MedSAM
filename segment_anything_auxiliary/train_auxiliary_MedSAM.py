# 导入了一些库
from U2_Net.model import U2NET
from segment_anything_auxiliary.MedAuxiliarySAM import MedAuxiliarySAM
from utils.data_convert import mean_iou, compute_loss, build_dataloader_auxiliary
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim, nn
from segment_anything import sam_model_registry

print(torch.cuda.is_available())
# 获取GPU数量
gpu_count = torch.cuda.device_count()
print("Number of GPUs:", gpu_count)

# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1

bce_loss = nn.BCELoss(reduction='mean')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Thyroid', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help='')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=100, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models/', help='model path directory')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    parser.add_argument('--vit_type', type=str, default='vit_b', help='data directory')
    return parser.parse_known_args()[0]


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

    model_path = opt.model_path
    checkpoint = f".{model_path}{opt.dataset_name}_sam_best.pth"
    if not os.path.exists(checkpoint):
        checkpoint = '../work_dir/SAM/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[opt.vit_type](checkpoint=checkpoint)
    sam = sam.to(device)
    net = U2NET(3, 1)

    net = net.to(device)
    model = MedAuxiliarySAM(
        image_encoder=sam.image_encoder,
        mask_decoder=sam.mask_decoder,
        prompt_encoder=sam.prompt_encoder,
        u2net=net,
        device=device
    )

    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)

    optimizer = optim.AdamW(sam.mask_decoder.parameters(),
                            lr=lr, betas=beta, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_path = os.path.join(model_path,
                             f"{opt.dataset_name}_{opt.epochs}_{opt.batch_size}_" + str(len(os.listdir(model_path))))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Training Start')
    best_loss = 999999999

    best_mIOU = 0
    best_dice = 0

    tr_pl_loss_list = []
    tr_pl_mi_list = []
    tr_pl_dice_list = []
    val_pl_loss_list = []
    val_pl_mi_list = []
    val_pl_dice_list = []
    dataloaders = build_dataloader_auxiliary(opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers)
    for epoch in range(opt.epochs):
        train_loss_list = []
        train_miou_list = []
        train_dice_list = []
        model.train()
        iterations = tqdm(dataloaders['train'])

        # 循环进行模型的多轮训练
        for train_data in iterations:
            # 将训练数据移到指定设备，这里是GPU
            image_1024 = train_data['image_1024'].to(device, dtype=torch.float32)
            image_256 = train_data['image_256'].to(device, dtype=torch.float32)
            mask = train_data['mask'].to(device, dtype=torch.float32)
            # 对优化器的梯度进行归零
            optimizer.zero_grad()

            train_IOU, sam_pred, d0, d1, d2, d3, d4, d5, d6 = model(image_1024, image_256)

            # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
            train_true_iou, dice = mean_iou(sam_pred, mask, eps=1e-6)
            train_miou_list = train_miou_list + train_true_iou.tolist()
            train_dice_list = train_dice_list + dice.tolist()

            u2net_loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
            sam_loss = compute_loss(sam_pred, mask)

            train_loss_one = sam_loss + u2net_loss
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
            pbar_desc += f", total dice: {np.mean(train_dice_list):.5f}"
            iterations.set_description(pbar_desc)

        train_loss = np.mean(train_loss_list)
        train_miou = np.mean(train_miou_list)
        train_dice = np.mean(train_dice_list)

        torch.cuda.empty_cache()
        tr_pl_loss_list.append(train_loss)
        tr_pl_mi_list.append(train_miou)
        tr_pl_dice_list.append(train_dice)
        lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0:
            model_path1 = save_path + "/" + opt.dataset_name + "_sam_" + str(epoch + 1 + epoch_add) + '_' + str(
                round(lr, 10)) + '.pth'
            torch.save(model.state_dict(), model_path1)
        print("epoch : {:3d}, train loss : {:3.4f}, train mIOU : {:3.4f}, train dice: {:3.4f})"
              .format(epoch + 1 + epoch_add, train_loss, train_miou, train_dice))
        scheduler.step()
        #  ------------------------------eval-------------------------
        model.eval()
        print('val Start')
        with torch.no_grad():
            val_loss_list = []
            val_miou_list = []
            val_dice_list = []
            iterations = tqdm(dataloaders['val'])

            # 循环进行模型的多轮训练
            for val_data in iterations:
                # 将训练数据移到指定设备，这里是GPU
                image_1024 = val_data['image_1024'].to(device, dtype=torch.float32)
                image_256 = val_data['image_256'].to(device, dtype=torch.float32)
                mask = val_data['mask'].to(device, dtype=torch.float32)

                val_IOU, sam_pred, d0, d1, d2, d3, d4, d5, d6 = model(image_1024, image_256, False)

                # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
                val_true_iou, dice = mean_iou(sam_pred, mask, eps=1e-6)
                val_miou_list = val_miou_list + val_true_iou.tolist()
                val_dice_list = val_dice_list + dice.tolist()

                u2net_loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, mask)
                sam_loss = compute_loss(sam_pred, mask)

                val_loss_one = sam_loss + u2net_loss

                val_loss_list.append(val_loss_one.item())

                pbar_desc = "Model val loss --- "
                pbar_desc += f"Total loss: {np.mean(val_loss_list):.5f}"
                pbar_desc += f", total mIOU: {np.mean(val_miou_list):.5f}"
                pbar_desc += f", total dice: {np.mean(val_dice_list):.5f}"
                iterations.set_description(pbar_desc)

            val_loss = np.mean(val_loss_list)
            val_miou = np.mean(val_miou_list)
            val_dice = np.mean(val_dice_list)
            torch.cuda.empty_cache()
            val_pl_loss_list.append(val_loss)
            val_pl_mi_list.append(val_miou)
            val_pl_dice_list.append(val_dice)

            if best_mIOU < val_miou:
                best_loss = val_loss
                best_mIOU = val_miou
                best_dice = val_dice
                best_path = save_path + f'/{opt.dataset_name}_sam_best.pth'
                torch.save(model.state_dict(), best_path)
                model = model.to(device)
                f = open(os.path.join(save_path, 'best.txt'), 'w')
                f.write(f"Experimental Day: {datetime.now()}")
                f.write("\n")
                f.write(f"mIoU: {str(best_mIOU)}")
                f.write("\n")
                f.write(f"dice: {str(best_dice)}")
                f.write("\n")
                f.write(f"epochs:{opt.epochs}")
                f.write("\n")
                f.write(f"batch_size:{opt.batch_size}")
                f.write("\n")
                f.write(f"learning_rate:{opt.lr}")
                f.write("\n")
                f.write(f"Data_set : {opt.dataset_name}")
                f.close()

            print("epoch : {:3d}, val loss : {:3.4f}, val mIOU : {:3.4f}, best loss : {:3.4f}, best mIOU : {:3.4f}, "
                  "best dice : {:3.4f})"
                  .format(epoch + 1 + epoch_add, val_loss, val_miou, best_loss, best_mIOU, best_dice))

    # (2, 2) 形式的图使用matplotlib可视化训练进展，生成用于训练和验证平均IOU、训练和验证损失的图表。
    plt_dict = {
        "Train_mIoU": tr_pl_mi_list,
        "Train_Loss": tr_pl_loss_list,
        "Train_dice": tr_pl_dice_list,
        "val_mIoU": val_pl_mi_list,
        "val_Loss": val_pl_loss_list,
        "val_dice": val_pl_dice_list,
    }

    plt.figure(figsize=(15, 15))
    for i, (key, item) in enumerate(plt_dict.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(opt.epochs), item, label=f"{key}")
        plt.title(f"{key}", fontsize=20)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel(f'{key.split("_")[-1]}', fontsize=15)
        plt.grid(True)

    plt.savefig(save_path + f'/{opt.dataset_name}_sam_{opt.epochs}_{opt.batch_size}_{opt.lr}_result.png')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
