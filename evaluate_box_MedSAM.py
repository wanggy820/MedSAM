import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from MedSAM import MedSAM
from MyDatasets import MyDatasets
from U2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from U2_Net.model import U2NET # full size version 173.6 MB
from segment_anything import sam_model_registry
import logging
from utils.data_convert import getDatasets, find_u2net_bboxes
from torchvision.transforms import transforms

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


colors = [
    (255, 255, 255),
]

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dice_iou_function(pred, target, smooth=1.0):
    pred = pred.reshape(-1)/255
    target = target.reshape(-1)/255
    intersection = torch.sum(pred * target)
    total = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (total + smooth)
    iou = (intersection + smooth) / (total - intersection + smooth)
    return dice, iou

def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--datasets", type=str, default='MICCAI', help="datasets")
    parser.add_argument("--root_dir", type=str, default='./datasets/', help="root_dir")
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    return parser

def build_dataloader(sam, model_name, data_dir, batch_size, num_workers, bboxes):
    dataloaders = {}
    for key in ['train', 'test']:
        image_list, mask_list = getDatasets(model_name, data_dir, key)
        dataloaders[key] = DataLoader(
            MyDatasets(sam, image_list, mask_list, bboxes),
            batch_size=batch_size,
            shuffle=True if key != 'test' else False,
            num_workers=num_workers,
            pin_memory=False
        )
    return dataloaders

def main():
    opt = get_argparser().parse_args()

    datasets = opt.datasets
    image_list, mask_list = getDatasets(datasets, opt.root_dir, "test")
    # image_list = [image_list[0]]
    # mask_list = [mask_list[0]]
    print("Number of images: ", len(image_list))

    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_' + datasets + '.pth'

    logging.basicConfig(filename="./val/" + datasets + '_val' + '.log', filemode="w", encoding='utf-8', level=logging.DEBUG)
    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(image_list=image_list,
                                        mask_list=mask_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_loader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir,map_location=device))
    net.to(device)
    net.eval()

    prompt_bboxes = []
    prompt_masks = []
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            inferencing = image_list[index]
            inputs=data["image"]
            #####################################  U2Net
            inputs = inputs.type(torch.FloatTensor).to(device)
            d1, d2, d3, d4, d5, d6, d7 = net(inputs)

            box = find_u2net_bboxes(d1, inferencing)
            prompt_bboxes.append(box)
            prompt_masks.append(d1)

    # checkpoint = f"./models/{datasets}_sam_best.pth"
    checkpoint = "work_dir/MedSAM/medsam_vit_b.pth"
    # set up model
    sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    medsam_model = MedSAM(sam_model).to(device)
    medsam_model.train()

    datasets = MyDatasets(image_list, mask_list, prompt_bboxes, prompt_masks, data_type="test")
    medsam_loader = DataLoader(datasets, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        # --------- 4. inference for each image ---------
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, data in enumerate(medsam_loader):
            inferencing = data["image_path"]
            print("inferencing:", inferencing)
            logging.info("inferencing:{}".format(inferencing))
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

            pre_mask, _ = medsam_model(image, prompt_box, prompt_masks, height, width)




            pre_mask = pre_mask.to(device)
            interaction_iou, interaction_dice = dice_iou_function(pre_mask, true_mask)

            interaction_total_dice += interaction_dice
            interaction_total_iou += interaction_iou

            print("interaction iou:{:3.6f}, interaction dice:{:3.6f}".format(interaction_iou, interaction_dice))
            print("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                  .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))
            logging.info("interaction iou:{:3.6f}, interaction dice:{:3.6f}"
                         .format(interaction_iou, interaction_dice))
            logging.info("interaction mean iou:{:3.6f},interaction mean dice:{:3.6f}"
                         .format(interaction_total_iou / (index + 1), interaction_total_dice / (index + 1)))


if __name__ == "__main__":
    main()
