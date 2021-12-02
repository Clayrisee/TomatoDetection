import cv2
import numpy as np
import math

def convert_rgb_to_hsi(img):
    """
    Function for convert RGB color to HSI color
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.float(img) / 255
        
        red = rgb[:, :, 0]
        green = rgb[:, :, 1]
        blue = rgb[:, :, 2]

        # Calculate intesity

        def cal_intensity(red, blue, green):
            return np.divide(red + blue + green, 3)
        
        # Calculate saturation
        def cal_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 1e-3) * minimum)
            return saturation
        
        # Calculate hue
        def cal_hue(red, blue, green):
            hue = np.copy(red)
            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                        math.sqrt((red[i][j] - green[i][j])**2 + ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])
                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]
            
            return hue
        
        hsi = cv2.merge((cal_hue(red, blue, green), cal_saturation(red, blue, green), cal_intensity(red, blue, green)))
        return hsi

