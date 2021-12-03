import cv2
import numpy as np
import math

def convert_rgb_to_hsi(img):
    """
    Function for convert RGB color to HSI color
    return: HSI image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = img / 255.0
        
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
        h = cal_hue(red, blue, green)
        s = cal_saturation(red, blue, green)
        i = cal_intensity(red, blue, green)
        print(h)
        print(s)
        print(i)
        hsi = cv2.merge((cal_hue(red, blue, green), cal_saturation(red, blue, green), cal_intensity(red, blue, green)))
        return hsi


def normalize_layer(layer):
    layer_shape = layer.shape[:2]
    normalize_array = np.full(layer_shape, 255.0)
    return np.sum(layer) / np.sum(normalize_array)

def find_hue_value(r, g, b):
    a = ((r - g) + (r - b)) / 2
    b = math.sqrt(((r-g)**2) + ((r-b) * (g-b)))
    return math.acos((a / b))

def find_saturation_value(r, g, b):
    min_value = np.minimum(np.minimum(r, g), b)
    return 1 - ((3/(r+g+b)) * min_value)

def find_intensity_value(r, g, b):
    return (r+g+b) / 3

def find_hsi_value(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = normalize_layer(img[:,:,0])
    g = normalize_layer(img[:,:,1])
    b = normalize_layer(img[:,:,2])
    print(r, g, b)
    return (find_hue_value(r,g, b), find_saturation_value(r, g, b), find_intensity_value(r, g, b))
    