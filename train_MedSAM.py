# 导入了一些库
import warnings
warnings.filterwarnings(action='ignore')
import monai
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from MedSAM import MedSAM
from MyDatasets import MyDatasets
from U2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from U2_Net.model import U2NET
from utils.data_convert import getDatasets
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt
import argparse
import torch
from torch import optim, nn
from segment_anything import sam_model_registry
warnings.filterwarnings(action='ignore')


# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='MICCAI', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help=' ')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=20, help='train epcoh')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight_decay')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models_no_box/', help='model path directory')
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

    medsam_model = MedSAM(sam).to(device)
    medsam_model.train()

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=lr, weight_decay=opt.weight_decay
    )
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.CrossEntropyLoss()

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

    train_datasets = MyDatasets(image_list, mask_list, bboxes, masks, "train")
    train_dataloader = DataLoader(train_datasets, batch_size=opt.batch_size,
                                  shuffle=False, num_workers=opt.num_workers, pin_memory=False)

    test_datasets = MyDatasets(image_list, mask_list, bboxes, masks, "test")
    test_dataloader = DataLoader(test_datasets, batch_size=opt.batch_size,
                                 shuffle=False, num_workers=opt.num_workers, pin_memory=False)

    for epoch in range(opt.epochs):
        train_loss_list = []
        train_miou_list = []

        sam.train()
        iterations = tqdm(train_dataloader)

        # 循环进行模型的多轮训练
        for data in iterations:
            optimizer.zero_grad()
            # inferencing = data["image_path"]
            # print("inferencing:", inferencing)
            image = data["image"].to(device)
            height = data["height"].to(device)
            width = data["width"].to(device)
            if "mask" in data:
                true_mask = data["mask"].to(device)
            else:
                true_mask = None

            if "prompt_box" in data:
                prompt_box = data["prompt_box"].to(device)
            else:
                prompt_box = None

            if "prompt_masks" in data:
                prompt_masks = data["prompt_masks"].to(device)
            else:
                prompt_masks = None

            pre_mask, iou = medsam_model(image, prompt_box, prompt_masks, height, width)
            pre_mask = pre_mask.to(device)
            iou = iou.to(device)
            loss = (seg_loss(pre_mask, true_mask) + ce_loss(pre_mask/255, (true_mask/255).float()))

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_list.append(loss.item())
            train_miou_list = train_miou_list + iou.tolist()
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

        medsam_model.eval()

        with torch.no_grad():
            valid_loss_list = []
            valid_miou_list = []

            iterations = tqdm(test_dataloader)
            for data in iterations:
                image = data["image"].to(device)
                height = data["height"].to(device)
                width = data["width"].to(device)
                if "mask" in data:
                    true_mask = data["mask"].to(device)
                else:
                    true_mask = None

                if "prompt_box" in data:
                    prompt_box = data["prompt_box"].to(device)
                else:
                    prompt_box = None

                if "prompt_masks" in data:
                    prompt_masks = data["prompt_masks"].to(device)
                else:
                    prompt_masks = None

                pre_mask, iou = medsam_model(image, prompt_box, prompt_masks, height, width)
                pre_mask = pre_mask.to(device)
                iou = iou.to(device)

                loss = seg_loss(pre_mask, true_mask) + ce_loss(pre_mask/255, (true_mask/255).float())
                loss.requires_grad_(True)
                loss.backward()
                valid_miou_list = valid_miou_list + iou.tolist()

                valid_loss_list.append(loss.item())

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
            model_path = save_path + "/" + opt.datasets + "_sam_" + str(epoch + 1 + epoch_add) + '_' + str(
                round(lr, 10)) + '.pth'
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