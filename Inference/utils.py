import cv2
import numpy as np

def draw_bbox(img, prediction):
    OFFSET = 70
    for result in prediction['results']:
        x1, y1, x2, y2 = result['bounding_box'] 
        x1 += OFFSET
        x2 += OFFSET
        y1 -= OFFSET
        y2 -= OFFSET
        cv2.rectangle(img,(x1, y1),(x2, y2), (0,255,0), 2)
        cv2.putText(img, result['label'], (x1, y1-10), 0, 0.6, (0,255,0))
        # BUG: bbox is shifted to bottom left when getting plotted

def draw_predictions(imgL, imgL_calibrated, imgL_pred, imgR, imgR_calibrated, imgR_pred):
    draw_bbox(imgL_calibrated, imgL_pred)
    draw_bbox(imgR_calibrated, imgR_pred)
    imgL_concat = np.concatenate((imgL, imgL_calibrated), axis=1)
    imgR_concat = np.concatenate((imgR, imgR_calibrated), axis=1)
    img_concat = np.concatenate((imgL_concat, imgR_concat), axis=0)

    cv2.imwrite('predictions.jpg',img_concat)
