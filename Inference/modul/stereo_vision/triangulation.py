import sys
import numpy as np
import cv2

def find_depth(center_point_left, center_point_right, frame_left, frame_right, focal, alpha, baseline=6):
    """
    Function for calculate depth estimation from stereo camera
    """
    # convert focal length from mm to pixel
    h_right, w_right, c_right = frame_right.shape
    h_left, w_left, c_left = frame_left.shape

    if w_left == w_right :
        f_pixel = (w_right * 0.5) / np.tan(alpha * 0.5 * np.pi / 180)
    else:
        print("Left and Right camera do not have same pixel width")

    # got top left x value for each frame
    x_right = center_point_right[0]
    x_left = center_point_left[0]

    # calculate disparity
    disparity = x_left - x_right
    
    # calculate depth
    zDepth = (baseline * f_pixel) / disparity

    return zDepth

    # TODO 1: Tentuin returnan resultnya
    # TODO 2: Tentuin limitasi objek antar 2 gambar, just in case deteksinya beda antara gambar 1 dan gambar 2
    # TODO 3: Depthnya ditentuin dari sorted array image

