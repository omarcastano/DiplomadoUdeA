import tensorflow as tf
import numpy as np

def IoU(pred_box, true_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred =  tf.split(pred_box, 4, axis = 1)
    xmin_true, ymin_true, xmax_true, ymax_true = tf.split(true_box, 4, axis = 1)

    smoothing_factor = 1e-10

    xmin_overlap = tf.maximum(xmin_pred, xmin_true)
    xmax_overlap = tf.minimum(xmax_pred, xmax_true)
    ymin_overlap = tf.maximum(ymin_pred, ymin_true)
    ymax_overlap = tf.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = tf.maximum((xmax_overlap - xmin_overlap), 0)  * tf.maximum((ymax_overlap - ymin_overlap), 0)
    union_area = (pred_box_area + true_box_area) - overlap_area
    
    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)

    return iou
