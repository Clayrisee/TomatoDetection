from modul.tomato_detection.TomatoModel import TomatoModel
import pprint
import cv2

from modul.tomato_detection.tools import find_hsi_value
from modul.stereo_vision import calibration, object_pairing
from utils import draw_predictions

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx")

imgL = cv2.imread("test_image/sample_left2.jpeg", 1)
imgR = cv2.imread("test_image/sample_right2.jpeg", 1)

## Calibration
imgL = cv2.resize(imgL, (640,480), interpolation=cv2.INTER_LINEAR)
imgR = cv2.resize(imgR, (640,480), interpolation=cv2.INTER_LINEAR)
imgL_calibrated, imgR_calibrated = calibration.undistortRectify(imgL, imgR)

imgL_pred = model.predict(imgL_calibrated)
imgR_pred = model.predict(imgR_calibrated)

pprint.pprint(imgL_pred)
pprint.pprint(imgR_pred)

imgL_pred_paired, imgR_pred_paired = object_pairing.get_pair(imgL_pred, imgR_pred)

 # PREDICTION DEBUG
pprint.pprint(imgL_pred_paired)
draw_predictions(imgL, imgL_calibrated, imgL_pred_paired,
                 imgR, imgR_calibrated, imgR_pred_paired)
