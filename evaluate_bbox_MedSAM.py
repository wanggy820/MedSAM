import argparse
import torch
import os
from segment_anything import sam_model_registry
import logging
from utils.data_convert import build_dataloader_box, save_output, calculate_dice_iou
from torch.nn.functional import threshold, normalize


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--dataset_name", type=str, default='ISBI', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--use_box', type=bool, default=False, help='is use box')
    return parser

def main():
    opt = get_argparser().parse_args()

    dataset_name = opt.dataset_name

    logging.basicConfig(filename="./val/" + dataset_name + '_val' + '.log', filemode="w", encoding='utf-8', level=logging.DEBUG)
    pre_dataset = "./pre_" + dataset_name + "/"
    if not os.path.exists(pre_dataset):
        os.mkdir(pre_dataset)

    # --------- 3. model define ---------
    model_path = "./models_box/"
    if opt.use_box == False:
        model_path = "./models_no_box/"

    checkpoint = f"./{model_path}{dataset_name}_sam_best.pth"
    # set up model
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
    sam.eval()
    dataloaders = build_dataloader_box(sam, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers)
    with torch.no_grad():
        # --------- 4. inference for each image ---------
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, data in enumerate(dataloaders['val']):
            image_path = data["image_path"]
            print("image_path:", image_path)
            logging.info("image_path:{}".format(image_path))
            mask_path = data["mask_path"]
            # 将训练数据移到指定设备，这里是GPU
            test_input = data['image'].to(device)
            prompt_box = data["prompt_box"].to(device)
            prompt_masks = data["prompt_masks"].to(device)
            test_encode_feature = sam.image_encoder(test_input)

            if opt.use_box == True:
                test_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points=None, boxes=prompt_box,
                                                                                    masks=prompt_masks)
            else:
                test_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points=None, boxes=prompt_box,
                                                                                    masks=None)

            #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
            test_mask, test_IOU = sam.mask_decoder(
                image_embeddings=test_encode_feature,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=test_sparse_embeddings,
                dense_prompt_embeddings=train_dense_embeddings,
                multimask_output=False)

            ##################################### MEDSAM
            for iPath, mPath in zip(image_path, mask_path):
                arr = iPath.split("/")
                image_name = arr[len(arr) - 1]
                if image_name.find("\\"):
                    arr = image_name.split("\\")
                    image_name = arr[len(arr) - 1]
                save_image_name = pre_dataset + image_name

                preds = normalize(threshold(test_mask, 0.0, 0)).squeeze(1)
                save_output(preds, iPath, save_image_name)
                dice, iou = calculate_dice_iou(save_image_name, mPath)
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



if __name__ == "__main__":
    main()
