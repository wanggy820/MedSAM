import argparse
import numpy as np
import cv2
import torch
import os
from PIL import Image

from BPAT_UNet.our_model.BPATUNet_all import BPATUNet
from MySAMModel import MySAMModel
from MySegmentModel.modeling.MySegmentModel import build_model
from segment_anything import sam_model_registry
import logging
from utils.data_convert import mean_iou, build_dataloader
from TRFE_Net.visualization.metrics import Metrics, evaluate

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--dataset_name", type=str, default='Thyroid_tn3k', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    parser.add_argument('--save_models_path', type=str, default='./save_models', help='model path directory')
    parser.add_argument('--vit_type', type=str, default='vit_b', help='sam vit type')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('-auxiliary_model_path', type=str, default='../BPAT_UNet/BPAT-UNet_best.pth')
    return parser


def main():
    opt = get_argparser().parse_args()

    save_models_path = opt.save_models_path
    dataset_model = f"{save_models_path}/{opt.dataset_name}_fold{opt.fold}"
    prefix = f"{dataset_model}/{opt.vit_type}_{opt.ratio:.2f}"
    logging.basicConfig(filename=f'{prefix}/val.log', filemode="w", level=logging.DEBUG)
    val_dataset = f"{prefix}/val/"
    if not os.path.exists(val_dataset):
        os.mkdir(val_dataset)

    # --------- 3. model define ---------
    best_checkpoint = f"{prefix}/sam_best.pth"
    # set up model

    auxiliary_model = BPATUNet(n_classes=1)
    auxiliary_model.load_state_dict(torch.load(opt.auxiliary_model_path, weights_only=True))
    auxiliary_model = auxiliary_model.to(device)
    auxiliary_model.eval()

    model = build_model(checkout=best_checkpoint)
    model = model.to(device)
    model.eval()
    sam = sam_model_registry[opt.vit_type](checkpoint="../work_dir/SAM/sam_vit_b_01ec64.pth").to(device)
    dataloaders = build_dataloader(sam, auxiliary_model, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers, opt.ratio, opt.fold)
    with torch.no_grad():
        metrics = Metrics(['precision', 'recall', 'specificity', 'F1_score', 'auc', 'acc', 'iou', 'dice', 'mae', 'hd'])
        # --------- 4. inference for each image ---------
        interaction_total_dice = 0
        interaction_total_iou = 0
        dataloader = dataloaders['test']
        for index, data in enumerate(dataloader):
            image_path = data["image_path"]
            print(f"index:{index + 1}/{len(dataloader)},image_path:{image_path}")
            logging.info("image_path:{}".format(image_path))
            mask_path = data["mask_path"]
            # 将训练数据移到指定设备，这里是GPU
            mask = data['mask'].to(device, dtype=torch.float32)
            size = data["size"]
            unet_pre, mask_former, low_res_pred = model(data)

            _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(low_res_pred, mask)
            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)
            # res_pre = low_res_pred * 255
            iou, dice = mean_iou(low_res_pred, mask, eps=1e-6)
            iou = iou.item()
            dice = dice.item()
            interaction_total_dice += dice
            interaction_total_iou += iou
            print("interaction iou:{:3.6f}, interaction dice:{:3.6f}"
                  .format(iou, dice))
            print("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                  .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))
            logging.info("interaction iou:{:3.6f}, interaction dice:{:3.6f}"
                         .format(iou, dice))
            logging.info("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                         .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))

            res_pre = torch.where(low_res_pred > 0.5, 255.0, 0.0)
            # res_pre = low_res_pred * 255.0
            ##################################### MEDSAM
            for mPath, pre, (w, h) in zip(mask_path, res_pre, size):
                arr = mPath.split("/")
                image_name = arr[len(arr) - 1]
                if image_name.find("\\"):
                    arr = image_name.split("\\")
                    image_name = arr[len(arr) - 1]
                save_image_name = val_dataset + image_name
                if os.path.isfile(save_image_name):
                    os.remove(save_image_name)

                # 保存为灰度图
                # 保存为灰度图
                predict = pre.unsqueeze(0)

                height = h.item()
                width = w.item()
                if height > width:
                    height = sam.image_encoder.img_size
                    width = int(w.item()*sam.image_encoder.img_size/h.item())
                else:
                    width = sam.image_encoder.img_size
                    height = int(h.item()*sam.image_encoder.img_size/w.item())
                predict = sam.postprocess_masks(predict, (height, width),
                                                (h.item(), w.item()))
                predict = predict.squeeze()
                predict_np = predict.cpu().data.numpy()
                im = Image.fromarray(predict_np).convert('L')
                imo = im.resize((w.item(), h.item()), resample=Image.BILINEAR)

                # gray_image_cv = cv2.cvtColor(np.array(imo), cv2.COLOR_RGB2BGR)
                # smooth = cv2.medianBlur(gray_image_cv, 5)
                # smooth1 = (smooth > 127)*255
                # imo = Image.fromarray(np.uint8(smooth1)).convert('L')

                imo.save(save_image_name)

        metrics_result = metrics.mean(len(dataloader))
        DSC_new = (2 * metrics_result['iou']) / (1 + metrics_result['iou'])
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f, DSC: %.4f'
            % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
               metrics_result['F1_score'],
               metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
               metrics_result['hd'], metrics_result['auc'], DSC_new))

if __name__ == "__main__":
    main()
