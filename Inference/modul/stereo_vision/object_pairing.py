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
    
    check_input(bb1, bb2)
    intersection_area = get_intersection_area(bb1, bb2)
    bb1_area = get_area(bb1)
    bb2_area = get_area(bb2)
    iou = get_iou(intersection_area, bb1_area, bb2_area)
    check_output(iou)
    return iou

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

    if is_intersecting(x_left, x_right, y_top, y_bottom):
        return (x_right - x_left) * (y_bottom - y_top)
    return 0.0

def is_intersecting(x_left, x_right, y_top, y_bottom):
    return x_right < x_left or y_bottom < y_top

def get_iou(intersection_area, bb1_area, bb2_area):
    return intersection_area / float(bb1_area + bb2_area - intersection_area)

