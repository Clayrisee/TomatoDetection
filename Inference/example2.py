from modul.stereo_vision import calibration
from modul.tomato_detection.TomatoModel import TomatoModel
import cv2
import pprint

model = TomatoModel(onnx_path="model/tomato_detection_yolov4_tiny.onnx") # Load onnx model
img_left = cv2.imread("test_image/sample_left.jpeg", 1) # Input image
img_right = cv2.imread("test_image/sample_right.jpeg", 1)
img_left, img_right = calibration.undistortRectify(img_left, img_right)
final_result = model.predict_stereo_vision(img_left, img_right)

pprint.pprint(final_result)
