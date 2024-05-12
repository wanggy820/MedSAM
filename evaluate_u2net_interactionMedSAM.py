import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import transforms

from U2_Net.U2netSegDataset import U2netSegDataset
from U2_Net.data_loader import ToTensorLab, RescaleT
from U2_Net.data_loader import SalObjDataset
from U2_Net.model import U2NET # full size version 173.6 MB
from skimage import io
from segment_anything import sam_model_registry
import logging
from utils.data_convert import compute_sam_dice, getDatasets

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
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    iou = (intersection + smooth) / (pred_flat.sum() + target_flat.sum() - intersection + smooth)
    return dice, iou

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma - mi) / 2.0, 1.0, 0)
    return dn

def find_u2net_bboxes(input, image_name):
    # normalization
    pred = input[:, 0, :, :]
    masks = normPRED(pred)

    predict = masks.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pred = np.array(imo)
    gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    maxW = 0
    maxH = 0
    maxX = 0
    maxY = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x*h > maxW*maxH:
            maxX = x
            maxY = y
            maxW = w
            maxH = h

    return np.array([[maxX, maxY, maxX + maxW, maxY + maxH]])


def get_argparser():
    parser = argparse.ArgumentParser()
    # model Options
    parser.add_argument("--datasets", type=str, default='DRIVE', help="datasets")
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
    test_dataset = U2netSegDataset(image_list, mask_list, input_size=(320, 320))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir,map_location=device))
    net.to(device)
    net.eval()

    checkpoint = f"./models/{datasets}_sam_best.pth"
    # set up model
    medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)
    medsam_model.eval()

    with torch.no_grad():
        # --------- 4. inference for each image ---------
        u2net_total_dice = 0
        u2net_total_iou = 0
        interaction_total_dice = 0
        interaction_total_iou = 0
        for index, (inputs, labels) in enumerate(test_loader):
            inferencing = image_list[index]
            print("inferencing:", inferencing)
            logging.info("inferencing:{}".format(inferencing))

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
