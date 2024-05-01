# 导入了一些库
import warnings
from utils.data_convert import build_dataloader, mean_iou, compute_loss, dice_function
warnings.filterwarnings(action='ignore')
import numpy as np
from tqdm import tqdm
import argparse
import torch
from segment_anything import sam_model_registry


# 设置了一些配置参数
beta = [0.9, 0.999]
milestone = [60000, 86666]
gamma = 0.1

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ISBI', help='model name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--warmup_steps', type=int, default=250, help=' ')
    parser.add_argument('--global_step', type=int, default=0, help=' ')
    parser.add_argument('--epochs', type=int, default=1, help='train epcoh')
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

    checkpoint = f"./models/{opt.model_name}_sam_best.pth"
    sam = sam_model_registry['vit_b'](checkpoint=checkpoint)
    sam = sam.to(device=device)

    print('Training Start')

    dataloaders = build_dataloader(sam, opt.model_name, opt.data_dir, opt.batch_size, opt.num_workers)
    torch.cuda.empty_cache()

    sam.eval()
    with torch.no_grad():
        valid_loss_list = []
        valid_miou_list = []
        valid_dice_list = []
        iterations = tqdm(dataloaders['test'])

        for valid_data in iterations:
            valid_input = valid_data['image'].to(device)
            valid_target_mask = valid_data['mask'].to(device, dtype=torch.float32)

            valid_encode_feature = sam.image_encoder(valid_input)
            valid_sparse_embeddings, valid_dense_embeddings = sam.prompt_encoder(points=None, boxes=None,
                                                                                 masks=None)

            valid_mask, valid_IOU = sam.mask_decoder(image_embeddings=valid_encode_feature,

                                                     image_pe=sam.prompt_encoder.get_dense_pe(),

                                                     sparse_prompt_embeddings=valid_sparse_embeddings,

                                                     dense_prompt_embeddings=valid_dense_embeddings,

                                                     multimask_output=False)

            valid_true_iou = mean_iou(valid_mask, valid_target_mask, eps=1e-6)
            valid_miou_list = valid_miou_list + valid_true_iou.tolist()

            valid_loss_one = compute_loss(valid_mask, valid_target_mask, valid_IOU, valid_true_iou)
            valid_loss_list.append(valid_loss_one.item())

            valid_dice_one = dice_function(valid_mask, valid_target_mask)
            valid_dice_list.append(valid_dice_one.item())

            pbar_desc = f"{opt.model_name} valid loss ---"
            pbar_desc += f" mean loss: {np.mean(valid_loss_list):.6f}"
            pbar_desc += f", mean mIOU: {np.mean(valid_miou_list):.6f}"
            pbar_desc += f", mean dice: {np.mean(valid_dice_list):.6f}"
            iterations.set_description(pbar_desc)

        valid_loss = np.mean(valid_loss_list)
        valid_miou = np.mean(valid_miou_list)
        valid_dice = np.mean(valid_dice_list)
        print("{} mean loss : {:3.6f}, mean mIOU : {:3.6f}, mean dice : {:3.6f}"
              .format(opt.model_name, valid_loss, valid_miou, valid_dice))
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
