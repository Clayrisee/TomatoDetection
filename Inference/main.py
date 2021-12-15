from modul.tomato_detection.TomatoModel import TomatoModel
import pprint
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

pprint.pprint(model.predict(imgL_calibrated))
pprint.pprint(model.predict(imgR_calibrated))

 # RESULT DEBUG
imgL_calibrated, imgR_calibrated = calibration.undistortRectify(imgL, imgR)
imgL_concat = np.concatenate((imgL, imgL_calibrated), axis=1)
imgR_concat = np.concatenate((imgR, imgR_calibrated), axis=1)
img_concat = np.concatenate((imgL_concat, imgR_concat), axis=0)

cv2.imwrite('calibration.jpg',img_concat)
