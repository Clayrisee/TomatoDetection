from modul.tomato_detection.TomatoModel import TomatoModel
import pprint
import cv2
import numpy as np

from modul.tomato_detection.tools import find_hsi_value
from modul.stereo_vision import calibration

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

imgL = cv2.imread("test_image/test_image.jpg", 1)
imgR = cv2.imread("test_image/sample_right.jpeg", 1)

## Calibration
imgL = cv2.resize(imgL, (640,480), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (640,480), interpolation=cv2.INTER_LINEAR)
imgL_calibrated, imgR_calibrated = calibration.undistortRectify(imgL, imgR)

imgL_pred = model.predict(imgL_calibrated)
imgR_pred = model.predict(imgR_calibrated)

pprint.pprint(imgL_pred)
pprint.pprint(imgR_pred)

 # RESULT DEBUG
def draw_bbox(img, prediction):
    for result in prediction['results']:
        x1, y1, x2, y2 = result['bounding_box'] 
        cv2.rectangle(img,(x1, y1),(x2, y2), (0,255,0), 2)
        cv2.putText(img, result['label'], (x1, y1-10), 0, 0.6, (0,255,0))
        # BUG: bbox is shifted to bottom left when getting plotted

draw_bbox(imgL_calibrated, imgL_pred)
draw_bbox(imgR_calibrated, imgR_pred)
imgL_concat = np.concatenate((imgL, imgL_calibrated), axis=1)
imgR_concat = np.concatenate((imgR, imgR_calibrated), axis=1)
img_concat = np.concatenate((imgL_concat, imgR_concat), axis=0)

cv2.imwrite('predictions.jpg',img_concat)
