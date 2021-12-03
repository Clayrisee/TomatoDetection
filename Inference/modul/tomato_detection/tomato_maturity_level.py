import cv2
import numpy as np
from .tools import convert_rgb_to_hsi, find_hsi_value

def calculate_tomato_maturity_level(img, bboxs):
    tomato_images = []
    hsi_values = []
    for bbox in bboxs:
        print(bbox)
        x1, y1, x2, y2 = bbox
        if x1 < 0 :
            x1 = 0
        elif x2 < 0:
            x2 = 0
        elif y1 < 0:
            y1 = 0
        elif y2 < 0:
            y2 = 0
        tomato_img = img[x1:x1+x2, y1:y1+y2]
        # print(type(tomato_img))
        # print(tomato_img.shape)
        # print(tomato_img)
        hsi_value = find_hsi_value(tomato_img)
        hsi_values.append(hsi_value)
    
    return hsi_values