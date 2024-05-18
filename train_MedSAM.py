# 导入了一些库
import warnings
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from MyDatasets import MyDatasets
from U2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from U2_Net.model import U2NET
from utils.data_convert import build_dataloader, mean_iou, compute_loss, find_u2net_bboxes, getDatasets
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim
from segment_anything import sam_model_registry
warnings.filterwarnings(action='ignore')


# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='DRIVE', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help=' ')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=500, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models/', help='model path directory')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--pretrained', type=str, default=False, help='pre trained model select')

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

    datasets = opt.datasets
    image_list, mask_list = getDatasets(datasets, opt.data_dir, "test")
    # image_list = [image_list[0]]
    # mask_list = [mask_list[0]]
    print("Number of images: ", len(image_list))

    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + datasets + '.pth'
    test_salobj_dataset = SalObjDataset(image_list=image_list,
                                        mask_list=mask_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_loader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=device))
    net.to(device)
    net.eval()

    bboxes = []
    masks = []
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            inferencing = image_list[index]
            inputs = data["image"]
            #####################################  U2Net
            inputs = inputs.type(torch.FloatTensor).to(device)
            d1, d2, d3, d4, d5, d6, d7 = net(inputs)

            box = find_u2net_bboxes(d1, inferencing)
            bboxes.append(box)
            masks.append(d1)


    checkpoint = f"./models/{opt.datasets}_sam_best.pth"
    if not os.path.exists(checkpoint):
        checkpoint = './work_dir/SAM/sam_vit_b_01ec64.pth'

    sam = sam_model_registry['vit_b'](checkpoint=checkpoint)

    if opt.pretrained:
        sam.load_state_dict(torch.load('./models/' + opt.pretrained))
        sam = sam.to(device=device)
    else:
        sam = sam.to(device=device)

    optimizer = optim.AdamW(sam.mask_decoder.parameters(),
                            lr=lr, betas=beta, weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

    # 脚本在各个检查点保存训练模型的状态字典，如果模型在验证集上取得最佳平均IOU，则单独保存最佳模型。
    if len(os.listdir(opt.model_path)) == 0:
        save_path = os.path.join(opt.model_path, f"{opt.datasets}_model_{opt.epochs}_{opt.batch_size}_0")
        os.makedirs(save_path)
    else:
        save_path = os.path.join(opt.model_path,
                                 f"{opt.datasets}_model_{opt.epochs}_{opt.batch_size}_" + str(len(os.listdir(opt.model_path))))
        os.makedirs(save_path)

    print('Training Start')
    best_loss = 999999999

    best_mIOU = 0

    tr_pl_loss_list = []
    tr_pl_mi_list = []
    val_pl_loss_list = []
    val_pl_mi_list = []

    mDatasets = MyDatasets(sam, image_list, mask_list, bboxes, masks, "train")
    dataloader = DataLoader(mDatasets, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=False)
    for epoch in range(opt.epochs):
        train_loss_list = []
        train_miou_list = []

        sam.train()
        iterations = tqdm(dataloader)
        # 循环进行模型的多轮训练
        for train_data in iterations:
            # 将训练数据移到指定设备，这里是GPU
            train_input = train_data['image'].to(device)

            train_target_mask = train_data['mask'].to(device, dtype=torch.float32)
            train_IOU = train_data['iou']
            # 对优化器的梯度进行归零
            optimizer.zero_grad()

            # 计算预测IOU和真实IOU之间的差异，并将其添加到列表中。然后计算训练损失（总损失包括mask损失和IOU损失），进行反向传播和优化器更新。
            train_true_iou = mean_iou(train_input, train_target_mask, eps=1e-6)
            train_miou_list = train_miou_list + train_true_iou.tolist()

            train_loss_one = compute_loss(train_input, train_target_mask, train_IOU, train_true_iou)
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

        with torch.no_grad():
            valid_loss_list = []
            valid_miou_list = []

            iterations = tqdm(dataloader)
            for valid_data in iterations:
                valid_input = valid_data['image'].to(device)
                valid_target_mask = valid_data['mask'].to(device, dtype=torch.float32)
                valid_IOU = train_data['iou']

                valid_true_iou = mean_iou(valid_input, valid_target_mask, eps=1e-6)
                valid_miou_list = valid_miou_list + valid_true_iou.tolist()

                valid_loss_one = compute_loss(valid_input, valid_target_mask, valid_IOU, valid_true_iou)
                valid_loss_list.append(valid_loss_one.item())

                pbar_desc = "Model valid loss --- "
                pbar_desc += f"Total loss: {np.mean(valid_loss_list):.5f}"
                pbar_desc += f", total mIOU: {np.mean(valid_miou_list):.5f}"
                iterations.set_description(pbar_desc)

            valid_loss = np.mean(valid_loss_list)
            valid_miou = np.mean(valid_miou_list)
            val_pl_loss_list.append(valid_loss)
            val_pl_mi_list.append(valid_miou)

        sam = sam.to(device)
        if best_mIOU < valid_miou:
            best_loss = valid_loss
            best_mIOU = valid_miou
            model_path = save_path + f'/{opt.datasets}_sam_best.pth'
            sam = sam.cpu()
            torch.save(sam.state_dict(), model_path)
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
            f.write(f"Data_set : {opt.data_dir}")
            f.close()

        print("epoch : {:3d}, train loss : {:3.4f}, valid loss : {:3.4f}, valid mIOU : {:3.4f}\
            ( best vaild loss : {:3.4f}, best valid mIOU : {:3.4f} )".format(epoch + 1 + epoch_add,
                                                                             train_loss,
                                                                             valid_loss,
                                                                             valid_miou,
                                                                             best_loss,
                                                                             best_mIOU
                                                                             ))

        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 5 == 0 or (epoch + 1) in [1, 2, 3, 4, 5]:
            model_path = save_path + "/" +opt.datasets + "_sam_" + str(epoch + 1 + epoch_add) + '_' + str(round(lr, 10)) + '.pth'
            sam = sam.cpu()
            torch.save(sam.state_dict(), model_path)
            sam = sam.to(device)

    # (2, 2) 形式的图使用matplotlib可视化训练进展，生成用于训练和验证平均IOU、训练和验证损失的图表。
    plt_dict = {
        "Train_mIoU": tr_pl_mi_list,
        "Val_mIoU": val_pl_mi_list,
        "Train_Loss": tr_pl_loss_list,
        "Val_Loss": val_pl_loss_list
    }

    plt.figure(figsize=(15, 15))
    for i, (key, item) in enumerate(plt_dict.items()):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(opt.epochs), item, label=f"{key}")
        plt.title(f"{key}", fontsize=20)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel(f'{key.split("_")[-1]}', fontsize=15)
        plt.grid(True)

    plt.savefig(save_path + f'/{opt.datasets}_sam_{opt.epochs}_{opt.batch_size}_{opt.lr}_result.png')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)