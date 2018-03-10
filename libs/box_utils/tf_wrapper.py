# -*- coding: utf-8 -*_


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import cv2

from libs.box_utils.iou_cpu import get_iou_matrix
from libs.box_utils.nms_gpu import rotate_gpu_nms
from libs.box_utils.iou_gpu import rbbx_overlaps

def get_iou_matrix_tf(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''

    :param boxes_list1:[N, 5] tensor
    :param boxes_list2: [M, 5] tensor
    :return:
    '''

    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)
    if use_gpu:
        iou_matrix = tf.py_func(rbbx_overlaps,
                                inp=[boxes1, boxes2, gpu_id],
                                Tout=tf.float32)
    else:
        iou_matrix = tf.py_func(get_iou_matrix, inp=[boxes1, boxes2],
                                Tout=tf.float32)

    iou_matrix = tf.reshape(iou_matrix, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return iou_matrix


def nms_rotate_tf(boxes_list, scores, iou_threshold, max_output_size, use_gpu=True, gpu_id=0):

    if use_gpu:
        det_tensor = tf.concat([boxes_list, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(rotate_gpu_nms,
                          inp=[det_tensor, iou_threshold, gpu_id],
                          Tout=tf.int64)
        keep = tf.cond(
            tf.greater(tf.shape(keep)[0], max_output_size),
            true_fn=lambda: tf.slice(keep, [0], [max_output_size]),
            false_fn=lambda: keep)
        keep = tf.reshape(keep, [-1])
        return keep
    else:
        raise NotImplementedError("not implemented the CPU vesion because of low speed")

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


#-------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------


def get_mask_tf(rotate_rects, featuremap_size):
    mask_tensor = tf.py_func(get_mask,
                            inp=[rotate_rects, featuremap_size],
                            Tout=tf.float32)
    mask_tensor = tf.reshape(mask_tensor, [tf.shape(rotate_rects)[0], featuremap_size, featuremap_size]) # [300, 14, 14]

    return mask_tensor


def get_mask(rotate_rects, featuremap_size):

    all_mask = []
    for a_rect in rotate_rects:
        rect = ((a_rect[1], a_rect[0]), (a_rect[3], a_rect[2]), a_rect[-1])  # in tf. [x, y, w, h, theta]
        rect_eight = cv2.boxPoints(rect)
        x_list = rect_eight[:, 0:1]
        y_list = rect_eight[:, 1:2]
        min_x, max_x = np.min(x_list), np.max(x_list)
        min_y, max_y = np.min(y_list), np.max(y_list)
        x_list = x_list - min_x
        y_list = y_list - min_y

        new_rect = np.hstack([x_list*featuremap_size*1.0/(max_x-min_x+1),
                             y_list * featuremap_size * 1.0 / (max_y - min_y + 1)])
        mask_array = np.zeros([featuremap_size, featuremap_size], dtype=np.float32)
        for x in range(featuremap_size):
            for y in range(featuremap_size):
                inner_rect = cv2.pointPolygonTest(contour=new_rect, pt=(x, y), measureDist=False)
                mask_array[y, x] = np.float32(0) if inner_rect == -1 else np.float32(1)
        all_mask.append(mask_array)
    return np.array(all_mask)
