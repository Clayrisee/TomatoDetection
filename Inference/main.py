from modul.tomato_detection.TomatoModel import TomatoModel
import cv2
import numpy as np

from modul.tomato_detection.tools import find_hsi_value
from modul.stereo_vision import calibration

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

img = cv2.imread("test_image/test_image_2.jpg", 1)
# imgL = cv2.imread("test_image/sample_left.jpeg", 1)
# imgR = cv2.imread("test_image/sample_right.jpeg", 1)

## Calibration
# imgL = cv2.resize(imgL, (640,480), interpolation=cv2.INTER_LINEAR)
# imgR = cv2.resize(imgR, (640,480), interpolation=cv2.INTER_LINEAR)
# print(imgL.shape)
# print(imgR.shape)

# imgL, imgR = calibration.undistortRectify(imgL, imgR)

# hsi_value = find_hsi_value(img)
# print(hsi_value)
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# red = rgb[:, :, 0]
# green = rgb[:, :, 1]
# blue = rgb[:, :, 2]
# print(red)
# sum_red = np.sum(red)
# print(sum_red)
# print(sum_red/ 255)
# Test Predict


model.predict(img)
# left_result_coors, left_labels = model.predict(imgL)
# right_result_coors, right_labels = model.predict(imgR)
