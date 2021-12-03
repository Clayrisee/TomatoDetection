# Tomato Detection Inference File
Tomato Detection using YoloV4 Models (running in onnxruntime). This model will detect tomato and count it from input images. The model will return dictionary results of prediction model.

# Installation

Clone the repository
```bash
git clone https://github.com/Clayrisee/TomatoDetection.git
```
Go to inference directory and run init_project.sh script
```bash
cd Inference
sh init_project.sh
```

Walaa~, you can detect the tomato.

# Directory Structure
```bash
├── init_project.sh # script for initialize project
├── main.py # main script for inference
├── model
│   └── yolov4_1_3_416_416_static.onnx # model
├── modul # our custom package for yolov4 inference
│   ├── stereo_vision # Stereo vision package for run yolov4 in stereo camera
│   │   ├── calibrate_camera.py
│   │   ├── calibration.py
│   │   ├── images
│   │   │   ├── stereoLeft
│   │   │   │   ├── imageL0.png
│   │   │   │   ├── imageL1.png
│   │   │   │   ├── imageL2.png
│   │   │   │   ├── imageL3.png
│   │   │   │   ├── imageL4.png
│   │   │   │   └── imageL5.png
│   │   │   └── stereoRight
│   │   │       ├── imageR0.png
│   │   │       ├── imageR1.png
│   │   │       ├── imageR2.png
│   │   │       ├── imageR3.png
│   │   │       ├── imageR4.png
│   │   │       └── imageR5.png
│   │   ├── __init__.py
│   │   ├── stereo_calibration.py
│   │   └── stereoMap.xml
│   └── tomato_detection # main modul for detection.
│       ├── __init__.py
│       ├── tomato_maturity_level.py
│       ├── TomatoModel.py
│       └── tools.py
├── README.md
├── requirements.txt
└── test_image
    ├── sample_left.jpeg
    ├── sample_right.jpeg
    ├── test_image_2.jpg
    └── test_image.jpg

```


# How to use
If you only want to use TomatoModel for detect object you can follow the example bellow.

## Initialization argument
```python
class TomatoModel:
    def __init__(self, onnx_path="model/yolov4_1_3_416_416_static.onnx", threshold=0.5 ,input_size=(416, 416)):
        """
        Parameters:
        ----------
        onnx_path: str
                Path onnx weights
        threshold: float
                Threshold for set minimum confidence of the model
        input_size: tuple
                Input Image Size
        """
```
## Output Format
```python
{
    "tomato_count": int,
    "results": [
        {
            "bounding_boxes": "{Return Tupple of Coordinates Prediction}",
            "label":"tomato",
            "confidence": "{Accuracy prediction model}"
        }
    ]
}
```
## Example
```python
from modul.tomato_detection.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="model/yolov4_1_3_416_416_static.onnx") # Load onnx model
img = cv2.imread("test_image/test_image_2.jpg", 1) # Input image
results = model.predict(img)
print(results)
# OUTPUT:
# {'tomato_count': 4, 'results': [{'bounding_box': (20, 301, 192, 411), 'label': 'tomato', 'confidence': 0.99486065}, {'bounding_box': (10, 109, 196, 235), 'label': 'tomato', 'confidence': 0.9945724}, {'bounding_box': (0, 0, 182, 84), 'label': 'tomato', 'confidence': 0.9843072}, {'bounding_box': (43, 205, 220, 300), 'label': 'tomato', 'confidence': 0.89212906}]}
```


# Feedback

If you have any feedback, please reach out to us at linkedin
  
## 🔗 Contact Us
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/haikalardikatama/)

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/manfredmichael/?locale=in_ID)

Clayrisee@2021
  
