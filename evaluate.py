import os
import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image
import glob
from U2_Net.data_loader import ToTensorLab
from U2_Net.data_loader import SalObjDataset
from U2_Net.model import U2NET # full size version 173.6 MB

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

# dice:0.9131831226826423, iou:0.8402364204687426  A-908.png
# 0.8855221741986402 IOU: 0.7945623983923332
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

def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2net


    image_dir = './datasets/MICCAI2023/val'
    prediction_dir = './predict_u2net_results'
    if(not os.path.exists(prediction_dir)):
        os.mkdir(prediction_dir)
    else:
        file_list = os.listdir(prediction_dir)
        # 遍历文件列表并删除每个文件
        for file in file_list:
            # 构建完整的文件路径
            file_path = os.path.join(prediction_dir, file)
            if os.path.isfile(file_path):
                # 如果是文件则直接删除
                os.remove(file_path)


    model_dir = './U2_Net/saved_models/u2net/u2net_bce_best_MICCAI.pth'

    img_name_list = glob.glob(image_dir+'/image/*')

    lbl_name_list = glob.glob(image_dir+'/mask/*')
    print("Number of images: ", len(img_name_list))

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=lbl_name_list,
                                        transform=ToTensorLab(flag=0)
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

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor).to(device)
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_test)
        # save results to test_results folder
        image_path = save_output(img_name_list[i_test], d1, prediction_dir)

        dice, iou = dice_iou(image_path, lbl_name_list[i_test])
        print("u2net----- dice:{}, iou:{}".format(dice, iou))


if __name__ == "__main__":
    main()
