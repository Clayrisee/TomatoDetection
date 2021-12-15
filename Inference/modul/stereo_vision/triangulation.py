import sys
import numpy as np
import cv2

def find_depth(center_point_left, center_point_right, width, focal, alpha, baseline=6):
    """
    Function for calculate depth estimation from stereo camera
    """

    f_pixel = (width * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)

    # got top left x value for each frame
    x_right = center_point_right[0]
    x_left = center_point_left[0]

    # calculate disparity
    disparity = x_left - x_right
    
    # calculate depth
    zDepth = (baseline * f_pixel) / disparity

    return zDepth

    # TODO 1: Tentuin returnan resultnya (DONE)
    # TODO 2: Tentuin limitasi objek antar 2 gambar, just in case deteksinya beda antara gambar 1 dan gambar 2 (DONE)
    # TODO 3: Depthnya ditentuin dari sorted array image (DONE)

