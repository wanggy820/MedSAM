import random
from PIL import Image
import cv2
import numpy as np


def find_bboxes(image, bbox_shift=20):
    pred = np.array(image)
    im = Image.fromarray(pred * 255).convert('RGB')
    pred = np.array(im)
    gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape[-2:]
    y_indices, x_indices = np.where(gray > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min = max(0, x_min - random.randint(0, bbox_shift))
    x_max = min(W, x_max + random.randint(0, bbox_shift))
    y_min = max(0, y_min - random.randint(0, bbox_shift))
    y_max = min(H, y_max + random.randint(0, bbox_shift))
    box_np = np.array([[x_min, y_min, x_max, y_max]])
    box_np = box_np.astype(np.int16)
    return box_np


    _boxes = []
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        if bbox_shift > 0:
            x_min = max(0, x_min - random.randint(0, bbox_shift))
            x_max = min(w, x_max + random.randint(0, bbox_shift))
            y_min = max(0, y_min - random.randint(0, bbox_shift))
            y_max = min(h, y_max + random.randint(0, bbox_shift))
        _boxes.append([x_min, y_min, x_max, y_max])

    boxes = []
    for i in range(0, len(_boxes)):
        isBigRect = True
        ibox = _boxes[i]
        for j in range(i + 1, len(_boxes)):
            jbox = _boxes[j]
            if ibox[0] > jbox[0] and ibox[1] > jbox[1] and ibox[2] < jbox[2] and ibox[3] < jbox[3]:
                isBigRect = False
        if isBigRect:
            boxes.append(ibox)
    return np.array(boxes)