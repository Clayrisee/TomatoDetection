from modul.tomato_detection.TomatoModel import TomatoModel
import cv2
import numpy as np

from modul.tomato_detection.tools import find_hsi_value
model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

img = cv2.imread("test_image/test_image_2.jpg", 1)
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