import cv2

from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import torchvision.ops as ops
import os
from torchvision.ops import box_convert
from groundingdino.util.inference import GroundingDINOVisualizer
from config import ConfigurationManager, DataConfig, ModelConfig

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list


def process_images(
        model,
        text_prompt,
        data_config,
        box_threshold=0.30,
        text_threshold=0.20
):
    visualizer = GroundingDINOVisualizer(save_dir="visualizations")

    total_iou = 0
    for img in os.listdir(data_config.val_dir):
        image_path=os.path.join(data_config.val_dir,img)
        image_source, image = load_image(image_path)

        mask_path = os.path.join("../datasets/Thyroid_Dataset/tn3k/test-mask", img)
        mask_source, mask = load_image(mask_path)
        gray = cv2.cvtColor(mask_source, cv2.COLOR_BGR2GRAY)
        # 二值化处理
        _, mask1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        true_boxes = []
        for contour in contours:
            bx, by, bw, bh = cv2.boundingRect(contour)
            true_boxes.append([bx, by, bw+bx, bh+by])
        mAP = visualizer.visualize_image(model,image,text_prompt,image_source,img,  true_boxes, box_th=box_threshold,txt_th=text_threshold)

        print(f"img:{img}, {mAP}")
        total_iou += mAP
        #boxes, logits, phrases = predict(
        #    model=model,
        #    image=image,
        #    caption=text_prompt,
        #    box_threshold=box_threshold,
        #    text_threshold=text_threshold
        #)
        #print(f"Original boxes size {boxes.shape}")
        #if boxes.shape[0]>0:
        #    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
        #    print(f"NMS boxes size {boxes.shape}")
        #annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        #cv2.imwrite(f"vis_Dataset/{img}", annotated_frame)
    print(f"mean_average_precision:{total_iou/len(os.listdir(data_config.val_dir))}")
    # 0.679095447080131

if __name__ == "__main__":
    # Config file of the prediction, the model weights can be complete model weights but if use_lora is true then lora_wights should also be present see example
    ## config file
    config_path="configs/test_config.yaml"
    text_prompt="thyroid ."
    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,training_config.use_lora)
    process_images(model,text_prompt,data_config)
