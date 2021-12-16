import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    bb1 = convert_input_to_dict(bb1) 
    bb2 = convert_input_to_dict(bb2) 
    check_input(bb1, bb2)
    intersection_area = get_intersection_area(bb1, bb2)
    bb1_area = get_area(bb1)
    bb2_area = get_area(bb2)
    iou = get_iou_score(intersection_area, bb1_area, bb2_area)
    check_output(iou)
    return iou

def convert_input_to_dict(bb):
    return {'x1': bb[0],
            'y1': bb[1],
            'x2': bb[2],
            'y2': bb[3]}


def check_input(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

def check_output(iou):
    assert iou >= 0.0
    assert iou <= 1.0

def get_area(bb):
    return (bb['x2'] - bb['x1']) * (bb['y2'] - bb['y1'])

def get_intersection_area(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    
    # import pprint
    # pprint.pprint(bb1)
    # pprint.pprint(bb2)
    # print('x_left:', x_left)
    # print('y_top:', y_top)
    # print('x_right:', x_right)
    # print('y_bottom:', y_bottom)

    if is_intersecting(x_left, x_right, y_top, y_bottom):
        return (x_right - x_left) * (y_bottom - y_top)
    return 0.0

def is_intersecting(x_left, x_right, y_top, y_bottom):
    return x_right > x_left and y_bottom > y_top

def get_iou_score(intersection_area, bb1_area, bb2_area):
    return intersection_area / float(bb1_area + bb2_area - intersection_area)

def get_pair(imgL_pred, imgR_pred):
    # TODO: compare bbox properly from top left to bottom right to get pairs
    
    imgL_pred_paired = {'results':[]} 
    imgR_pred_paired = {'results':[]} 
    for i in range(len(imgL_pred['results'])):
        bbox_left = imgL_pred['results'][i]['bounding_box'] 
        iou_scores = []
        for j in range(len(imgR_pred['results'])):
            bbox_right = imgR_pred['results'][j]['bounding_box'] 
            iou_scores.append(get_iou(bbox_left, bbox_right))
       
        # print(iou_scores)
        if max(iou_scores) > 0.3:
            print(max(iou_scores))
            imgL_pred_paired['results'].append(imgL_pred['results'][i])
            imgR_pred_paired['results'].append(imgR_pred['results'][np.argmax(iou_scores)])
    return imgL_pred_paired, imgR_pred_paired
