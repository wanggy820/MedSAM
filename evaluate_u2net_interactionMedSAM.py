import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from U2_Net.data_loader import ToTensorLab, RescaleT
from U2_Net.data_loader import SalObjDataset
from U2_Net.model import U2NET  # full size version 173.6 MB
from segment_anything import sam_model_registry
import logging
from utils.data_convert import compute_sam_dice, getDatasets, find_u2net_bboxes

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

def dice_iou_function(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return dice, iou

def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--datasets", type=str, default='MICCAI', help="datasets")
    parser.add_argument("--root_dir", type=str, default='./datasets/', help="root_dir")
    return parser

def main():
    opt = get_argparser().parse_args()

    datasets = opt.datasets
    image_list, mask_list = getDatasets(datasets, opt.root_dir, "test")
    print("Number of images: ", len(image_list))


    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + datasets + '.pth'

    logging.basicConfig(filename="./val/" + datasets + '_val' + '.log', filemode="w", encoding='utf-8', level=logging.DEBUG)
    # --------- 2. dataloader ---------
    #1. dataloader
    test_dataset = SalObjDataset(image_list=image_list,
                                 mask_list=mask_list,
                                 transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir,map_location=device))
    net.to(device)
    net.eval()

    checkpoint = f"./models/{datasets}_sam_best.pth"
    checkpoint = "work_dir/MedSAM/medsam_vit_b.pth"
    # set up model
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
    medsam_model.eval()

    with torch.no_grad():
        # --------- 4. inference for each image ---------
        u2net_total_dice = 0
        u2net_total_iou = 0
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, data_test in enumerate(test_loader):
            inferencing = image_list[index]
            print("inferencing:", inferencing)
            logging.info("inferencing:{}".format(inferencing))
            inputs = data_test['image']
            labels = data_test['mask']
            #####################################  U2Net
            inputs = inputs.type(torch.FloatTensor).to(device)
            d1, d2, d3, d4, d5, d6, d7 = net(inputs)

            u2net_dice, u2net_iou = dice_iou_function(d1.cpu().numpy(), labels.cpu().numpy())
            u2net_total_dice += u2net_dice
            u2net_total_iou += u2net_iou

            ##################################### MEDSAM
            image_path = image_list[index]
            mask_path = mask_list[index]
            bboxes = find_u2net_bboxes(d1, image_path)
            valid_loss, interaction_iou, interaction_dice = compute_sam_dice(medsam_model, image_path, mask_path, bboxes)

            interaction_total_dice += interaction_dice
            interaction_total_iou += interaction_iou

            print("u2net       iou:{:3.6f}, u2net       dice:{:3.6f}".format(u2net_iou, u2net_dice))
            print("interaction iou:{:3.6f}, interaction dice:{:3.6f}".format(interaction_iou, interaction_dice))
            print("u2net       mean iou:{:3.6f},u2net       mean dice:{:3.6f}"
                  .format(u2net_total_iou / (index + 1), u2net_total_dice / (index + 1)))
            print("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                  .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))
            logging.info("u2net       iou:{:3.6f}, u2net       dice:{:3.6f}".format(u2net_iou, u2net_dice))
            logging.info("interaction iou:{:3.6f}, interaction dice:{:3.6f}"
                         .format(interaction_iou, interaction_dice))
            logging.info("u2net       mean iou:{:3.6f},u2net       mean dice:{:3.6f}"
                         .format(u2net_total_iou / (index + 1),u2net_total_dice / (index + 1)))
            logging.info("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                         .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))


if __name__ == "__main__":
    main()
