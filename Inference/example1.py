from modul.tomato_detection.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx") # Load onnx model
img = cv2.imread("test_image/test_image_2.jpg", 1) # Input image
results = model.predict(img)
print(results)
# OUTPUT:
# {'tomato_count': 4, 'results': [{'bounding_box': (20, 301, 192, 411), 'label': 'tomato', 'confidence': 0.99486065}, {'bounding_box': (10, 109, 196, 235), 'label': 'tomato', 'confidence': 0.9945724}, {'bounding_box': (0, 0, 182, 84), 'label': 'tomato', 'confidence': 0.9843072}, {'bounding_box': (43, 205, 220, 300), 'label': 'tomato', 'confidence': 0.89212906}]}