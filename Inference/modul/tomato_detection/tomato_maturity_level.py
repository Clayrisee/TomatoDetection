import cv2
import numpy as np
from .tools import convert_rgb_to_hsi, find_hsi_value, limit_bbox_coors

def calculate_tomato_maturity_level(img, bboxs):
    tomato_images = []
    hsi_values = []

    bboxs = limit_bbox_coors(bboxs, img)
    for bbox in bboxs:
        # print(bbox)
        x1, y1, x2, y2 = bbox 
        tomato_img = img[y1:y2, x1:x2]
        # print(type(tomato_img))
        # print(tomato_img.shape)
        # print(tomato_img)
        hsi_value = find_hsi_value(tomato_img)
        hsi_values.append(hsi_value)
    
    return hsi_values
