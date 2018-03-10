# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.remote_sensing_dict import LABEl_NAME_MAP
from libs.box_utils import boxes_utils
import os

def draw_box_with_color(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array([103.939, 116.779, 123.68])
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            # color = (50, 255, 0)
            cv2.drawContours(img, [rect], -1, color, 3)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


def draw_boxes_with_scores(img_batch, boxes, scores):

    def draw_box_cv(img, boxes, scores):
        img = img + np.array([103.939, 116.779, 123.68])
        boxes = boxes.astype(np.int64)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):

            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            score = scores[i]
            num_of_object += 1

            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            # rect = cv2.cv.BoxPoints(rect)
            rect = np.int0(rect)
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (50, 255, 0)
            cv2.drawContours(img, [rect], -1, color, 2)

            cv2.rectangle(img,
                          pt1=(x_c, y_c),
                          pt2=(x_c+50, y_c+15),
                          color=color,
                          thickness=-1)

            cv2.putText(img,
                        text=str(score)[:5],
                        org=(x_c, y_c+10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(color[0], color[2], color[1]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        img = img + np.array([103.939, 116.779, 123.68])
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            # ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1
                rect = ((x_c, y_c), (w, h), theta)
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                color = (50, 255, 0)
                cv2.drawContours(img, [rect], -1, color, 3)

                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c+120, y_c+15),
                              color=color,
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score)[:5],
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[0], color[2], color[1]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    img = cv2.imread('1.jpg')
    img = tf.constant(np.expand_dims(img, 0), tf.float32)
    boxes = tf.constant([[130, 130, 130, 130, -45]])
    labels = tf.constant([1])
    scores = tf.constant([0.6])
    img_ten = draw_boxes_with_categories(img, boxes, labels, scores)

    with tf.Session() as sess:
        img_np = sess.run(img_ten)
        # img_np = np.squeeze(img_np, 0)
        # cv2.imshow('test', img_np[0])
        # cv2.waitKey(0)
        cv2.imwrite('2.jpg', img_np[0][:, :, ::-1])