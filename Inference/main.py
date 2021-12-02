from modul.tomato_detection.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

img = cv2.imread("test_image/tomato37_png.rf.37c3c302d77c36ad7eb626901ec008cf.jpg")

# Test Predict
model.predict(img)