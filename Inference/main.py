from modul.tomato_detection.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

img = cv2.imread("test_image/test_image.jpg")

# Test Predict
model.predict(img)