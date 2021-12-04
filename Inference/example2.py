from modul.stereo_vision import calibration
from modul.tomato_detection.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx") # Load onnx model
img_left = cv2.imread("test_image/sample_left.jpeg", 1) # Input image
img_right = cv2.imread("test_image/sample_right.jpeg", 1)
img_left, img_right = calibration.undistortRectify(img_left, img_right)
results_left = model.predict(img_left)
result_right = model.predict(img_right)

print(results_left)
print(result_right)