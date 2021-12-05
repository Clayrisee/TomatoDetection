from modul.tomato_detection.TomatoModel import TomatoModel
import cv2
import numpy as np

from modul.tomato_detection.tools import find_hsi_value
from modul.stereo_vision import calibration

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

# img = cv2.imread("test_image/test_image_2.jpg", 1)
imgL = cv2.imread("test_image/sample_left2.jpeg", 1)
imgR = cv2.imread("test_image/sample_right2.jpeg", 1)
## Calibration
imgL = cv2.resize(imgL, (640,480), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (640,480), interpolation=cv2.INTER_LINEAR)
print(imgL.shape)
print(imgR.shape)

imgL, imgR = calibration.undistortRectify(imgL, imgR)


# print(model.predict(img))
left_results = model.predict(imgL)
right_results = model.predict(imgR)

import pprint
pprint.pprint(left_results)
pprint.pprint(right_results)
