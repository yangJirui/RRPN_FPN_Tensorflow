# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def forward_convert(coordinate, with_label=False):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            boxes.append(np.reshape(box, [-1, ]))

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinates, with_label=True):
    """
    :param coordinates: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    range of theta is  [-90, 0)
    """

    boxes = []
    if with_label:
        for rect in coordinates:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])
    else:
        for rect in coordinates:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)



if __name__ == '__main__':
    coord = np.array([[150, 150, 50, 100, -90],
                      [150, 150, 100, 50, -90],
                      [150, 150, 50, 100, -45],
                      [150, 150, 100, 50, -45]])

    coord1 = np.array([[150, 150, 100, 50, 0, 1],
                      [150, 150, 100, 50, -90, 1],
                      [150, 150, 100, 50, 45, 1],
                      [150, 150, 100, 50, -45, 1]])

    coord2 = forward_convert(coord1, True)
    print(coord2)
