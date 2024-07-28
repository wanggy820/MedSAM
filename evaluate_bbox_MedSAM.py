import argparse
import torch
import os
from PIL import Image
from segment_anything import sam_model_registry
import logging
from utils.data_convert import build_dataloader_box, calculate_dice_iou, mean_iou
from TRFE_Net.visualization.metrics import Metrics, evaluate

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--dataset_name", type=str, default='Thyroid_tg3k', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--model_path', type=str, default='./save_models', help='model path directory')
    parser.add_argument('--vit_type', type=str, default='vit_h', help='sam vit type')
    parser.add_argument('--prompt_type', type=int, default=3, help='0: None,1: box,2: mask,3: box and mask')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
    return parser


def main():
    opt = get_argparser().parse_args()

    model_path = opt.model_path
    dataset_model = f"{model_path}/{opt.dataset_name}"
    prefix = f"{dataset_model}/{opt.vit_type}_{opt.prompt_type}_{opt.ratio:.2f}"
    logging.basicConfig(filename=f'{prefix}/val.log', filemode="w", level=logging.DEBUG)
    val_dataset = f"{prefix}/val/"
    if not os.path.exists(val_dataset):
        os.mkdir(val_dataset)

    # --------- 3. model define ---------
    best_checkpoint = f"{prefix}/sam_best.pth"
    # set up model
    sam = sam_model_registry[opt.vit_type](checkpoint=best_checkpoint).to(device)
    sam.eval()
    dataloaders = build_dataloader_box(sam, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers, opt.ratio)
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
            test_input = data['image'].to(device)
            prompt_box = data["prompt_box"].to(device)
            prompt_masks = data["prompt_masks"].to(device)
            mask = data['mask'].to(device, dtype=torch.float32)
            size = data["size"]
            test_encode_feature = sam.image_encoder(test_input)

            if opt.prompt_type == 1:
                val_sparse_embeddings, val_dense_embeddings = sam.prompt_encoder(points=None,
                                                                                 boxes=prompt_box,
                                                                                 masks=None)
            elif opt.prompt_type == 2:
                val_sparse_embeddings, val_dense_embeddings = sam.prompt_encoder(points=None, boxes=None,
                                                                                 masks=prompt_masks)
            elif opt.prompt_type == 3:
                val_sparse_embeddings, val_dense_embeddings = sam.prompt_encoder(points=None,
                                                                                 boxes=prompt_box,
                                                                                 masks=prompt_masks)
            else:
                val_sparse_embeddings, val_dense_embeddings = sam.prompt_encoder(points=None, boxes=None,
                                                                                 masks=None)

            #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
            test_mask, test_IOU = sam.mask_decoder(
                image_embeddings=test_encode_feature,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=val_sparse_embeddings,
                dense_prompt_embeddings=val_dense_embeddings,
                multimask_output=False)

            low_res_pred = torch.sigmoid(test_mask)

            _precision, _recall, _specificity, _f1, _auc, _acc, _iou, _dice, _mae, _hd = evaluate(low_res_pred, mask)
            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1_score=_f1, acc=_acc, iou=_iou, mae=_mae, dice=_dice, hd=_hd, auc=_auc)
            # res_pre = low_res_pred * 255
            iou, dice = mean_iou(low_res_pred, mask, eps=1e-6)
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
                imo.save(save_image_name)

        metrics_result = metrics.mean(len(dataloader))
        print("Test Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1_score:%.4f, acc: %.4f, iou: %.4f, mae: %.4f, dice: %.4f, hd: %.4f, auc: %.4f'
            % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
               metrics_result['F1_score'],
               metrics_result['acc'], metrics_result['iou'], metrics_result['mae'], metrics_result['dice'],
               metrics_result['hd'], metrics_result['auc']))

if __name__ == "__main__":
    main()
