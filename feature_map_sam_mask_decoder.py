import torch.nn as nn

from BPAT_UNet.our_model.BPATUNet_all import BPATUNet
from BPAT_UNet.utils import SoftDiceLoss
from MedSAM import MedSAM
from skimage import transform, io
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import argparse
import os
import cv2
import numpy as np
import torch


from segment_anything.modeling import TwoWayTransformer, Sam, ImageEncoderViT, PromptEncoder
from segment_anything.modeling.mask_decoder_feature_map import FeatureMapMaskDecoder
from utils.box import find_bboxes
from utils.data_convert import getDatasets, build_dataloader, get_click_prompt
from functools import partial
from pathlib import Path
import urllib.request

device = torch.device("cpu")
# freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"



class MySAMModel(nn.Module):
    def __init__( self, sam, auxiliary_model):
        super().__init__()
        self.sam = sam
        self.auxiliary_model = auxiliary_model
        for param in sam.image_encoder.parameters():
            param.requires_grad = False
        self.data = None

    def forward(self, data):
        data = self.data
        image = data['image'].to(self.sam.device)
        prompt_box = data["prompt_box"].to(self.sam.device)
        prompt_masks = data["prompt_masks"].to(self.sam.device)
        points = get_click_prompt(data, self.sam.device)

        with torch.no_grad():
            encode_feature = self.sam.image_encoder(image)  # (3, 256, 64, 64)
            # 使用 sam 模型的 image_encoder 提取图像特征，并使用 prompt_encoder 提取稀疏和密集的嵌入。在本代码中进行提示输入，所以都是None.
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=points, boxes=prompt_box, masks=prompt_masks)
        #  通过 mask_decoder 解码器生成训练集的预测掩码和IOU
        pre_mask = self.sam.mask_decoder(
            image_embeddings=encode_feature,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False)
        low_res_pred = torch.sigmoid(pre_mask)
        return low_res_pred

    def setData(self, data):
        self.data = data


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=FeatureMapMaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        sam.load_state_dict(state_dict)
    return sam


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

class SAMTarget(nn.Module):
    def __init__(self, input):
        super(SAMTarget, self).__init__()

        self.input = input

    def forward(self, x):
        # 读取图片，将图片转为RGB

        crop_img = self.input["mask"]

        bce_loss = SoftDiceLoss()
        loss = bce_loss(x, crop_img)
        return loss


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='Thyroid_tn3k', help="dataset name")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='data directory')
    parser.add_argument('--save_models_path', type=str, default='./save_models', help='model path directory')
    parser.add_argument('--vit_type', type=str, default='vit_h', help='sam vit type')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio')
    parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument('-auxiliar_model', type=str, default='BPATUNet')
    parser.add_argument('-auxiliary_model', type=str, default='MySAMModel')
    parser.add_argument('-auxiliary_model_path', type=str, default='./BPAT_UNet/BPAT-UNet_best.pth')
    return parser


def main():
    opt = get_argparser().parse_args()
    # set up model

    save_models_path = opt.save_models_path
    dataset_model = f"{save_models_path}/{opt.dataset_name}_fold{opt.fold}"
    prefix = f"{dataset_model}/{opt.vit_type}_{opt.ratio:.2f}_heigh"

    # --------- 3. model define ---------
    best_checkpoint = f"{prefix}/sam_best.pth"

    checkpoint = './work_dir/SAM/sam_vit_b_01ec64.pth'
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint).to(device)

    medsam = MedSAM(sam.image_encoder, sam.mask_decoder, sam.prompt_encoder)
    medsam.eval()

    auxiliary_model = BPATUNet(n_classes=1)
    auxiliary_model.load_state_dict(torch.load(opt.auxiliary_model_path, map_location=torch.device('cpu')))
    auxiliary_model = auxiliary_model.to(device)
    auxiliary_model.eval()

    myModel = torch.load(best_checkpoint, map_location=torch.device('cpu'), weights_only=True)
    myModel = myModel.to(device)
    myModel.train()

    print(medsam.mask_decoder)
    target_layers = [myModel.sam.mask_decoder]

    img_name_list, lbl_name_list, _ = getDatasets(opt.dataset_name, opt.data_dir, "test", 0)
    dataloaders = build_dataloader(sam, auxiliary_model, opt.dataset_name, opt.data_dir, opt.batch_size,
                                   opt.num_workers, opt.ratio, opt.fold)

    # 实例化cam，得到指定feature map的可视化数据
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=myModel, target_layers=target_layers)

    for index, data in enumerate(dataloaders["test"]):
        image_path = data["image_path"][0]
        # mask_path = data["mask_path"]
        print(f"image_path:{image_path}")
        myModel.setData(data)
        net_input = data["image"]
        grayscale_cam = cam(net_input, targets=[SAMTarget(data)])
        grayscale_cam = grayscale_cam[0, :]


        img_np = io.imread(image_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np
        H, W, _ = img_3c.shape

        canvas_img = cv2.cvtColor(img_3c, cv2.COLOR_RGB2BGR)
        origin_cam = cv2.resize(grayscale_cam, (W, H))
        # 将feature map与原图叠加并可视化
        src_img = np.float32(canvas_img) / 255
        visualization_img = show_cam_on_image(src_img, origin_cam, use_rgb=False)

        arr = image_path.split("/")
        image_name = arr[len(arr) - 1]
        path = "./feature_map"
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path += os.sep + opt.dataset_name
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path += os.sep + image_name

        io.imsave(path, visualization_img)

        # cv2.imshow('feature map', visualization_img)
        # cv2.waitKey(0)




if __name__ == "__main__":
    main()
