import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
import glob

from torchvision.transforms import transforms

from U2_Net.data_loader import ToTensorLab, RescaleT
from U2_Net.data_loader import SalObjDataset
from U2_Net.model import U2NET # full size version 173.6 MB
from skimage import transform, io
from segment_anything import sam_model_registry
from torch.nn import functional as F


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024
prediction_dir = './predict_u2net_results'
interaction_dir = './interaction_u2net_results'
# set up model
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    dn = torch.where(dn > (ma - mi) / 2.0, (ma - mi), 0)
    return dn

def save_output(image_name, pred, d_dir):
    pred = normPRED(pred)
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    image_path = d_dir+'/'+imidx+'.png'
    im.save(image_path)
    return image_path

def dice_iou(pred, target, smooth=1e-5):
    # 读取并转换图像为二值化形式
    image1 = cv2.imread(pred, 0)
    c = image1.sum().item()
    image2 = cv2.imread(target, 0)
    _, image1_binary = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    _, image2_binary = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    # 计算交集和并集
    intersection = cv2.bitwise_and(image1_binary, image2_binary)
    union = cv2.addWeighted(image1_binary, 0.5, image2_binary, 0.5, 0)

    # 计算DICE系数
    num_pixels_intersecting = cv2.countNonZero(intersection)
    num_pixels_total = cv2.countNonZero(union)
    dice_coefficient = (2 * num_pixels_intersecting+smooth) / float(num_pixels_total + num_pixels_intersecting+smooth)
    iou_coefficient = (num_pixels_intersecting + smooth) / float(num_pixels_total+smooth)
    return dice_coefficient, iou_coefficient

colors = [
    (255, 255, 255),
]

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

    masks = np.expand_dims(pred, axis=0)
    boxes = []
    maxw = maxh = 0
    #
    for i in range(masks.shape[0]):
        mask = masks[i]
        coor = np.nonzero(mask)
        xmin = coor[0][0]
        xmax = coor[0][-1]
        coor[1].sort()  # 直接改变原数组，没有返回值
        ymin = coor[1][0]
        ymax = coor[1][-1]

        width = ymax - ymin
        height = xmax - xmin

        # 这儿可以不要，这是为了找到最大值方便分割成统一的矩形
        if width > maxw:
            maxw = width
        if height > maxh:
            maxh = height
        boxes.append((ymin, xmin, maxw, maxh))
    return np.array(boxes)

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

@torch.no_grad()
def get_embeddings(img_3c):
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    # if self.embedding is None:
    with torch.no_grad():
        embedding = medsam_model.image_encoder(
            img_1024_tensor
        )  # (1, 256, 64, 64)
        return embedding

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
def interaction_u2net_predict(bboxes, file_path, save_dir):
    box = [0, 0, 0, 0]
    box = np.array(box)

    for j in bboxes:
        if box[2] * box[3] < j[2] * j[3]:
            box = j
        else:
            continue
        xmin = j[0]
        ymin = j[1]
        xmax = j[0] + j[2]
        ymax = j[1] + j[3]

    img_np = io.imread(file_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    embedding = get_embeddings(img_3c)
    H, W, _ = img_3c.shape
    box_np = np.array([[xmin, ymin, xmax, ymax]])
    # print("bounding box:", box_np)
    box_1024 = box_np / np.array([W, H, W, H]) * 1024

    sam_mask = medsam_inference(medsam_model, embedding, box_1024, H, W)

    mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")
    mask_c[sam_mask != 0] = colors[0]

    aaa = file_path.split("/")
    image_path = save_dir + '/' + aaa[len(aaa) - 1]
    io.imsave(image_path, mask_c)
    return image_path

def main():
    create_clear_dir(prediction_dir)
    create_clear_dir(interaction_dir)

    model_name = "IBCI"
    if model_name == "MICCAI":
        model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_MICCAI.pth'
        image_dir = './datasets/MICCAI2023/val'
        img_name_list = glob.glob(image_dir + '/image/*')
        lbl_name_list = glob.glob(image_dir + '/mask/*')
    else:
        model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_ICBI.pth'

        datasets_dir = "./DeepLabV3Plus/datasets/"
        filePath = datasets_dir + "ISBI2016_ISIC_Part3B_Test_GroundTruth.csv"
        f = open(filePath, encoding="utf-8")
        data = pd.read_csv(f)
        img_name_list = []
        lbl_name_list = []

        for img, seg in zip(data["img"], data["seg"]):
            img_name_list.append(datasets_dir + img)
            lbl_name_list.append(datasets_dir + seg)



    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------

    print("...load U2NET---176.4 MB")
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_dir))
    net.to(device)
    net.eval()
    with torch.no_grad():
        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):
            print("inferencing:", img_name_list[i_test].split(os.sep)[-1])
            inputs_test = data_test['image']

            inputs_test = inputs_test.type(torch.FloatTensor).to(device)
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_test)
            # save results to test_results folder
            file_path = img_name_list[i_test]

            # u2net 图片保存
            u2net_image_path = save_output(file_path, d1, prediction_dir)
            dice, iou = dice_iou(u2net_image_path, lbl_name_list[i_test])

            bboxes = find_u2net_bboxes(d1, img_name_list[i_test])
            interaction_image_path = interaction_u2net_predict(bboxes, file_path, interaction_dir)
            dice1, iou1 = dice_iou(interaction_image_path, lbl_name_list[i_test])

            print("u2net----- dice:{}, iou:{}".format(dice, iou))
            print("interaction----- dice:{}, iou:{}".format(dice1, iou1))

if __name__ == "__main__":
    main()
