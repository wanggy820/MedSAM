# 导入了一些库
import json
import warnings
from BPAT_UNet.our_model.BPATUNet_all import BPATUNet
from MySegmentModel.modeling.MySegmentModel import build_model
from utils.data_convert import mean_iou, compute_loss, build_dataloader
warnings.filterwarnings(action='ignore')
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim
from TRFE_Net.visualization.metrics import Metrics, evaluate
from segment_anything import sam_model_registry


# 设置了一些配置参数
beta = (0.9, 0.999)
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Thyroid_tn3k', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help='')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=100, help='train epcoh')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    parser.add_argument('--save_models_path', type=str, default='./save_models', help='model path directory')
    parser.add_argument('--vit_type', type=str, default='vit_b', help='sam vit type')
    parser.add_argument('--ratio', type=float, default=1.00, help='ratio')
    parser.add_argument('-fold', type=int, default=0)
    parser.add_argument('-auxiliary_model_path', type=str, default='../BPAT_UNet/BPAT-UNet_best.pth')
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

    save_models_path = opt.save_models_path
    if not os.path.exists(save_models_path):
        os.makedirs(save_models_path)
    dataset_model = f"{save_models_path}/{opt.dataset_name}_fold{opt.fold}"
    if not os.path.exists(dataset_model):
        os.makedirs(dataset_model)
    prefix = f"{dataset_model}/{opt.vit_type}_{opt.ratio:.2f}"
    if not os.path.exists(prefix):
        os.makedirs(prefix)



    current_checkpoint = f"{prefix}/sam_current.pth"
    best_checkpoint = f"{prefix}/sam_best.pth"

    tr_pl_loss_list = []
    tr_pl_miou_list = []
    tr_pl_dice_list = []
    val_pl_loss_list = []
    val_pl_miou_list = []
    val_pl_dice_list = []
    best_mIOU = 0
    best_dice = 0
    start = 0
    # if os.path.exists(current_checkpoint):
    #     state_dict = torch.load(current_checkpoint, map_location=torch.device('cpu'))
    #     sam.load_state_dict(state_dict["model"])
    #     tr_pl_loss_list = state_dict["tr_pl_loss_list"]
    #     tr_pl_miou_list = state_dict["tr_pl_miou_list"]
    #     tr_pl_dice_list = state_dict["tr_pl_dice_list"]
    #     val_pl_loss_list = state_dict["val_pl_loss_list"]
    #     val_pl_miou_list = state_dict["val_pl_miou_list"]
    #     val_pl_dice_list = state_dict["val_pl_dice_list"]
    #
    #     best_mIOU = max(val_pl_miou_list)
    #     best_dice = max(val_pl_dice_list)
    #     start = state_dict["start"]



    auxiliary_model = BPATUNet(n_classes=1)
    auxiliary_model.load_state_dict(torch.load(opt.auxiliary_model_path))
    auxiliary_model = auxiliary_model.to(device)
    auxiliary_model.eval()

    model = build_model(checkout=best_checkpoint)
    model = model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=beta, weight_decay=opt.weight_decay)

    print('Training Start')

    sam = sam_model_registry[opt.vit_type](checkpoint="../work_dir/SAM/sam_vit_b_01ec64.pth").to(device)
    dataloaders = build_dataloader(sam, auxiliary_model, opt.dataset_name, opt.data_dir, opt.batch_size,
                                   opt.num_workers, opt.ratio, opt.fold)
    for epoch in range(start, opt.epochs):
        train_loss_list = []
        train_miou_list = []
        train_dice_list = []
        # -------------- train --------------
        model.train()
        iterations = tqdm(dataloaders['train'])

        # 循环进行模型的多轮训练
        for train_data in iterations:
            train_target_mask = train_data['mask'].to(device, dtype=torch.float32)
            # 对优化器的梯度进行归零
            optimizer.zero_grad()

            unet_pre, mask_former, low_res_pred = model(train_data)
            # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
            train_true_iou, train_true_dice = mean_iou(low_res_pred, train_target_mask, eps=1e-6)
            train_miou_list = train_miou_list + train_true_iou.tolist()
            train_dice_list = train_dice_list + train_true_dice.tolist()

            train_loss_one = (compute_loss(unet_pre, train_target_mask) +
                              compute_loss(mask_former, train_target_mask) +
                              compute_loss(low_res_pred, train_target_mask))
            train_loss_one.backward()

            optimizer.step()
            train_loss_list.append(train_loss_one.item())


            pbar_desc = "Model train loss --- "
            pbar_desc += f"Total loss: {np.mean(train_loss_list):.5f}"
            pbar_desc += f", total mIOU: {np.mean(train_miou_list):.5f}"
            pbar_desc += f", total dice: {np.mean(train_dice_list):.5f}"
            iterations.set_description(pbar_desc)


        train_loss = np.mean(train_loss_list)
        train_miou = np.mean(train_miou_list)
        train_dice = np.mean(train_dice_list)

        if best_dice < train_dice:
            best_mIOU = train_miou
            best_dice = train_dice
            torch.save(model.state_dict(), best_checkpoint)
            f = open(os.path.join(prefix, 'best.txt'), 'w')
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
            f.write(f"vit_type:{opt.vit_type}")
            f.write("\n")
            f.write(f"ratio:{opt.ratio}")
            f.write("\n")
            f.write(f"data_set : {opt.dataset_name}")
            f.close()

        torch.cuda.empty_cache()
        tr_pl_loss_list.append(train_loss)
        tr_pl_miou_list.append(train_miou)
        tr_pl_dice_list.append(train_dice)

        # -------------- eval --------------
        model.eval()
        val_loss_list = []
        val_miou_list = []
        val_dice_list = []
        with torch.no_grad():
            iterations = tqdm(dataloaders['test'])
            metrics = Metrics(
                ['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])

            # 循环进行模型的多轮训练
            for val_data in iterations:
                val_target_mask = val_data['mask'].to(device, dtype=torch.float32)
                unet_pre, mask_former, low_res_pred = model(val_data)

                # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
                val_true_iou, val_true_dice = mean_iou(low_res_pred, val_target_mask, eps=1e-6)
                val_miou_list = val_miou_list + val_true_iou.tolist()
                val_dice_list = val_dice_list + val_true_dice.tolist()

                val_loss_one = (compute_loss(unet_pre, val_target_mask) +
                                  compute_loss(mask_former, val_target_mask) +
                                  compute_loss(low_res_pred, val_target_mask))

                _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(low_res_pred, val_target_mask)
                metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                               F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)

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
            val_pl_miou_list.append(val_miou)
            val_pl_dice_list.append(val_dice)


        print("val epoch:{:3d},val mIOU:{:3.4f}, val dice:{:3.4f}, train best mIOU: {:3.4f}, train best dice: {:3.4f})"
              .format(epoch + 1 + epoch_add, val_miou, val_dice, best_mIOU, best_dice))

        metrics_result = metrics.mean(len(dataloaders['test']))
        print(
            'recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
            % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
               metrics_result['F1_score'],
               metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
               metrics_result['hd'], metrics_result['auc']))

        state_dict = {"tr_pl_loss_list": tr_pl_loss_list,
                      "tr_pl_miou_list": tr_pl_miou_list,
                      "tr_pl_dice_list": tr_pl_dice_list,
                      "val_pl_loss_list": val_pl_loss_list,
                      "val_pl_miou_list": val_pl_miou_list,
                      "val_pl_dice_list": val_pl_dice_list,
                      "model": model.state_dict(),
                      "start": epoch+1}
        torch.save(state_dict, current_checkpoint)

    # (2, 2) 形式的图使用matplotlib可视化训练进展，生成用于训练和验证平均IOU、训练和验证损失的图表。
    plt_dict = {
        "Train_Loss": tr_pl_loss_list,
        "Train_mIoU": tr_pl_miou_list,
        "Train_dice": tr_pl_dice_list,
        "val_Loss": val_pl_loss_list,
        "val_mIoU": val_pl_miou_list,
        "val_dice": val_pl_dice_list,
    }

    plt.figure(figsize=(20, 20))
    for i, (key, item) in enumerate(plt_dict.items()):
        plt.subplot(3, 2, i + 1)
        plt.plot(range(opt.epochs), item, label=f"{key}")
        plt.title(f"{key}", fontsize=16)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel(f'{key.split("_")[-1]}', fontsize=15)
        plt.grid(True)

    plt.savefig(f'{prefix}/result_lr_{opt.lr}.png')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
