# 导入了一些库
import os
import warnings
import logging
from segment_anything_u2net.build_u2net_sam import build_sam
from utils.data_convert import build_dataloader_box, calculate_dice_iou
import argparse
import torch
from PIL import Image

warnings.filterwarnings(action='ignore')

# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='MICCAI', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--model_path', type=str, default='./models_box/', help='model path directory')
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='data directory')
    parser.add_argument('--data_type', type=str, default='val', help='data directory')
    return parser.parse_known_args()[0]

def create_clear_dir(dir):
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    else:
        file_list = os.listdir(dir)
        # 遍历文件列表并删除每个文件
        for file in file_list:
            # 构建完整的文件路径
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                # 如果是文件则直接删除
                os.remove(file_path)

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

    model_path = "./models_no_box/"
    checkpoint = f"{model_path}{opt.dataset_name}_sam_best.pth"

    dataset_name = opt.dataset_name
    logging.basicConfig(filename=f'./val/{dataset_name }_val.log', encoding='utf-8', level=logging.DEBUG)

    sam = build_sam(checkpoint=checkpoint)
    sam = sam.to(device=device)
    sam.eval()

    print(f"starting {opt.data_type}")
    dataloaders = build_dataloader_box(sam, opt.dataset_name, opt.data_dir, opt.batch_size, opt.num_workers)

    interaction_dir = f'./val/{opt.dataset_name}'
    create_clear_dir(interaction_dir)

    with torch.no_grad():
        # --------- 4. inference for each image ---------
        interaction_total_dice = 0
        interaction_total_iou = 0
        dataloader = dataloaders[opt.data_type]
        for index, data in enumerate(dataloader):
            image_path = data["image_path"]
            print(f"index:{index + 1}/{len(dataloader)},image_path:{image_path}")
            logging.info("image_path:{}".format(image_path))
            mask_path = data["mask_path"]
            # 将训练数据移到指定设备，这里是GPU
            test_input = data['image'].to(device)
            prompt_box = data["prompt_box"].to(device)
            prompt_masks = data["prompt_masks"].to(device)
            size = data["size"]
            test_encode_feature = sam.image_encoder(test_input)

            if opt.use_box:
                test_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points=None, boxes=prompt_box,
                                                                                    masks=None)
            else:
                test_sparse_embeddings, train_dense_embeddings = sam.prompt_encoder(points=None, boxes=None,
                                                                                    masks=prompt_masks)

            #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
            test_mask, test_IOU = sam.mask_decoder(
                image_embeddings=test_encode_feature,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=test_sparse_embeddings,
                dense_prompt_embeddings=train_dense_embeddings,
                multimask_output=False)

            low_res_pred = torch.sigmoid(test_mask)

            res_pre = torch.where(low_res_pred > 0.5, 255.0, 0)
            ##################################### MEDSAM
            for mPath, pre, (w, h) in zip(mask_path, res_pre, size):
                arr = mPath.split("/")
                image_name = arr[len(arr) - 1]
                if image_name.find("\\"):
                    arr = image_name.split("\\")
                    image_name = arr[len(arr) - 1]
                save_image_name = interaction_dir + os.sep + image_name

                # 保存为灰度图
                predict = pre.unsqueeze(0)

                height = h.item()
                width = w.item()
                if height > width:
                    height = sam.image_encoder.img_size
                    width = int(w.item() * sam.image_encoder.img_size / h.item())
                else:
                    width = sam.image_encoder.img_size
                    height = int(h.item() * sam.image_encoder.img_size / w.item())
                predict = sam.postprocess_masks(predict, (height, width),
                                                (h.item(), w.item()))
                predict = predict.squeeze()
                predict_np = predict.cpu().data.numpy()
                im = Image.fromarray(predict_np).convert('L')
                imo = im.resize((w.item(), h.item()), resample=Image.BILINEAR)
                imo.save(save_image_name)

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


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)