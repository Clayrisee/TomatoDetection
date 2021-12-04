import cv2
import numpy as np
import onnxruntime as ort

from modul.stereo_vision.triangulation import find_depth

from .tools import find_centroid, match_bboxs, sorted_bboxs
from .tomato_maturity_level import *

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

        self.class_names = ["tomato"]
        self.input_size = input_size
        self.onnx_path = onnx_path
        self.threshold = threshold
        self.nms_threshold = (
            0
            if self.threshold - 0.1 < 0  
            else self.threshold - 0.1

        )

        self.__get_ort_session()

    
    def __get_ort_session(self):
        if ort.get_device() == "CPU":
            self.ort_session =  ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        else:
            self.ort_session =  ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
    
    def __preprocessing_img(self, img):
        w, h = img.shape[:2]
        if w > self.input_size[0] or h > self.input_size[1]:
            img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        # convert to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img /= 255.0
        return img

    def __nmsbbox(self, bbox, max_confidence, min_mode=False):
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = max_confidence.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= self.nms_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def __postprocessing_onnx(self, output_onnx):
        box_array = output_onnx[0]
        confs = output_onnx[1]
        num_classes = confs.shape[2]
        box_array = box_array[:, :, 0]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        bboxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.threshold

            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self.__nmsbbox(ll_box_array, ll_max_conf, self.nms_threshold)
                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )
            bboxes_batch.append(bboxes)

        return bboxes_batch
    
    def __postprocess_result_single_img(self, postprocess_onnx, width, height):
        # print(postprocess_onnx)
        result_coors = []
        # labels = []
        results = {
            "tomato_count": 0,
            "results" : []
        }

        for x1, y1, x2, y2, _, conf, label in postprocess_onnx[0]:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            bbox = tuple((x1, y1, x2, y2))
            bbox = tuple(0 if i<0 else i for i in bbox)
            result_item={
                "bounding_box": bbox,
                "label":self.class_names[label],
                "confidence":conf
            }
            results["results"].append(result_item)
            result_coors.append(bbox)
            # labels.append(f"{self.class_names[label]} conf: {conf}")
        if len(results["results"]) != 0:
            results["tomato_count"] = len(results["results"])
        return result_coors, results
    
    def __unpack_result_postprocess_onnx(self, postprocess_onnx, shape_img):
        result_coors = []
        labels = []
        confidences = []
        width, height = shape_img
        for x1, y1, x2, y2, _, conf, label in postprocess_onnx:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            bbox = tuple((x1, y1, x2, y2))
            bbox = tuple(0 if i<0 else i for i in bbox)
            result_coors.append(bbox)
            labels.append(self.class_names[label])
            confidences.append(conf)
        return result_coors, labels, confidences

    def __postprocess_result_stereo_vision(self, postporcess_onnx_left, postprocess_onnx_right, shape_img_left, shape_img_right):
        depth_estimation_results = []
        sorted_onnx_results_left = sorted_bboxs(postporcess_onnx_left[0])
        sorted_onnx_results_right = sorted_bboxs(postprocess_onnx_right[0])
        result_coors_left, result_labels_left, confs_left = self.__unpack_result_postprocess_onnx(sorted_onnx_results_left, shape_img_left)
        result_coors_right, result_labels_right, confs_right = self.__unpack_result_postprocess_onnx(sorted_onnx_results_right, shape_img_right)
        max_item, results_match_bboxs = match_bboxs(result_coors_left, result_coors_right)
        result_labels_left, result_labels_right = result_labels_left[:max_item], result_labels_right[:max_item]
        confs_left, confs_right = confs_left[:max_item], confs_right[:max_item]
        results_left, results_right = results_match_bboxs
        bboxs_left, centroids_left = results_left
        bboxs_right, centroids_right = results_right
        for centroid_left, centroid_right in zip(centroids_left, centroids_right):
            zDepth = find_depth(centroid_left, centroid_right, shape_img_left[0], focal=6, alpha = 56.6, baseline=6)
            depth_estimation_results.append(zDepth)
        pass

    def predict_stereo_vision(self, img_left, img_right):
        w_left, h_left = img_left.shape[:2]
        w_right, h_right = img_right.shape[:2]
        input_onnx = self.ort_session.get_inputs()[0].name
        if (w_left == w_right) and (h_left == h_right):
            # Process image
            img_left_preprocessed = self.__preprocessing_img(img_left)
            img_right_prepreocesed = self.__preprocessing_img(img_right)
            output_onnx_left = self.ort_session.run(None, {input_onnx: img_left_preprocessed})
            output_onnx_right = self.ort_session.run(None, {input_onnx:img_right_prepreocesed})
            output_onnx_left = self.__postprocessing_onnx(output_onnx_left)
            output_onnx_right = self.__postprocessing_onnx(output_onnx_right)
        else:
            raise ValueError("input size of image left must equal as image right!")


    def predict(self, img):
        widht_ori, height_ori = img.shape[:2]
        preprocess_img = self.__preprocessing_img(img)
        input_onnx = self.ort_session.get_inputs()[0].name
        output_onnx = self.ort_session.run(None, {input_onnx: preprocess_img})
        postprocess_onnx = self.__postprocessing_onnx(output_onnx)
        # print("POST PROCESS ONNX")
        # print(postprocess_onnx)
        result_coors, results = self.__postprocess_result_single_img(postprocess_onnx, widht_ori, height_ori)
        centroid_arrays = find_centroid(result_coors)
        centroid_arrays_sorted = sorted_bboxs(centroid_arrays)
        # hsi_values = calculate_tomato_maturity_level(img, result_coors)
        # print(centroid_arrays)
        print(centroid_arrays_sorted)
        # print(sorted_bboxs(result_coors))
        # print("Posprocess onnx:", postprocess_onnx)
        # print("Coors :", result_coors)
        # print("label_outputs:", labels)
        # print("hsi_values", hsi_values)
        
        return results
