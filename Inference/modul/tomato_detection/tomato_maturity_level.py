import cv2
import numpy as np
from .converter import convert_rgb_to_hsi

def calculate_tomato_maturity_level(img, bboxs):
    tomato_images = []
    hsi_values = []
    for bbox in bboxs:
        x1, y1, x2, y2 = bbox
        tomato_img = img[x1:x1+x2, y1:y1+y2]
        hsi_color = convert_rgb_to_hsi(tomato_img)
        hsi_values.append(hsi_color)
    
    return hsi_values