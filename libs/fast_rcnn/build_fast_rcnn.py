# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.box_utils import encode_and_decode
from libs.box_utils import tf_wrapper

from libs.losses import losses
from libs.configs import cfgs
from libs.box_utils.boxes_utils import get_horizon_minAreaRectangle, clip_boxes_to_img_boundaries
from libs.box_utils.show_box_in_tensor import *

# from libs.tf_utils.tf_ops import get_mask_tf

DEBUG = False


class FastRCNN(object):
    def __init__(self,
                 feature_pyramid, rpn_proposals_boxes, rpn_proposals_scores,
                 img_batch,
                 img_shape,
                 roi_size,
                 scale_factors,
                 roi_pool_kernel_size,  # roi size = initial_crop_size / max_pool_kernel size
                 gtboxes_and_label,  # [M, 5]
                 fast_rcnn_nms_iou_threshold,
                 fast_rcnn_maximum_boxes_per_img,
                 fast_rcnn_nms_max_boxes_per_class,
                 show_detections_score_threshold,  # show box scores larger than this threshold

                 num_classes,  # exclude background
                 fast_rcnn_minibatch_size,
                 fast_rcnn_positives_ratio,
                 fast_rcnn_positives_iou_threshold,
                 use_dropout,
                 is_training,
                 weight_decay,
                 stop_gradient_for_proposals,
                 top_k_nms,
                 nms_angle_threshold,
                 use_angle_condition,
                 level,
                 boxes_angle_threshold):

        self.feature_pyramid = feature_pyramid

        if stop_gradient_for_proposals:
            self.rpn_proposals_boxes = tf.stop_gradient(tf.reshape(rpn_proposals_boxes, [-1, 5]))  # [N, 5]
            self.rpn_proposals_scores = tf.stop_gradient(tf.reshape(rpn_proposals_scores, [-1, 1]))
        else:
            self.rpn_proposals_boxes = tf.reshape(rpn_proposals_boxes, [-1, 5])  # [N, 5]
            self.rpn_proposals_scores = rpn_proposals_scores

        if is_training and cfgs.ADD_GTBOXES_AS_PROPOSALS:
            self.rpn_proposals_boxes, self.rpn_proposals_scores = \
                self._add_gtboxes_as_first_stage_proposals(self.rpn_proposals_boxes, self.rpn_proposals_scores,
                                                           gtboxes_and_label[:, :-1])
        self.img_shape = img_shape
        self.img_batch = img_batch
        self.roi_size = roi_size
        self.roi_pool_kernel_size = roi_pool_kernel_size
        self.top_k_nms = top_k_nms
        self.nms_angle_threshold = nms_angle_threshold
        self.use_angle_condition = use_angle_condition
        self.level = level
        self.min_level = int(level[0][1])
        self.max_level = min(int(level[-1][1]), 5)
        self.boxes_angle_threshold = boxes_angle_threshold

        self.fast_rcnn_nms_iou_threshold = fast_rcnn_nms_iou_threshold
        self.fast_rcnn_nms_max_boxes_per_class = fast_rcnn_nms_max_boxes_per_class
        self.fast_rcnn_maximum_boxes_per_img = fast_rcnn_maximum_boxes_per_img
        self.show_detections_score_threshold = show_detections_score_threshold

        self.scale_factors = scale_factors

        # iou >= 0.5 is positive, others are negative
        self.fast_rcnn_positives_iou_threshold = fast_rcnn_positives_iou_threshold

        self.fast_rcnn_minibatch_size = fast_rcnn_minibatch_size
        self.fast_rcnn_positives_ratio = fast_rcnn_positives_ratio

        self.gtboxes_and_label = gtboxes_and_label
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.is_training = is_training
        self.weight_decay = weight_decay

        self.fast_rcnn_all_level_rois, self.fast_rcnn_all_level_rotate_proposals, \
        self.fast_rcnn_all_level_horizontal_proposals = self.get_rois()

        self.fast_rcnn_encode_boxes, self.fast_rcnn_scores = self.fast_rcnn_net()

    def assign_level(self):
        with tf.name_scope('assign_levels'):

            _, _, w, h, _ = tf.unstack(self.rpn_proposals_boxes, axis=1)
            w = tf.maximum(w, 0.)  # avoid w is negative
            h = tf.maximum(h, 0.)  # avoid h is negative

            levels = tf.round(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  # 4 + log_2(***)

            levels = tf.maximum(levels, tf.ones_like(levels) * (np.float32(self.min_level)))  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * (np.float32(self.max_level)))  # level maximum is 5

            return tf.cast(levels, tf.int32)

    def _add_gtboxes_as_first_stage_proposals(self, first_stage_proposals, first_stage_scores, gtboxes):

        # 1. jitter gtboxes
        ws = gtboxes[:, 2]
        hs = gtboxes[:, 3]
        thetas = gtboxes[:, 4]

        hs_offset = (tf.random_normal(shape=tf.shape(hs)) - 0.5)*0.1*hs
        ws_offset = (tf.random_normal(shape=tf.shape(ws)) - 0.5)*0.1*ws
        thetas_offset = (tf.random_normal(shape=tf.shape(thetas)) - 0.5)*0.1*thetas
        hs = hs + hs_offset
        ws = ws + ws_offset
        thetas = thetas + thetas_offset

        new_boxes = tf.transpose(tf.stack([gtboxes[:, 0], gtboxes[:, 1], ws, hs, thetas], axis=0))

        # 2. get needed added gtboxes
        num_needed_add = tf.minimum(tf.cast(cfgs.FAST_RCNN_MINIBATCH_SIZE*cfgs.FAST_RCNN_POSITIVE_RATE*0.5, tf.int32),
                                    tf.shape(gtboxes)[0])
        added_boxes_indices = tf.random_shuffle(tf.range(start=0, limit=tf.shape(new_boxes)[0]))
        added_boxes_indices = tf.slice(added_boxes_indices, begin=[0], size=[num_needed_add])
        added_boxes = tf.gather(new_boxes, added_boxes_indices)

        # 3. add them
        all_boxes = tf.concat([first_stage_proposals, added_boxes], axis=0)
        all_scores = tf.concat([first_stage_scores,  tf.ones(shape=[tf.shape(added_boxes)[0]])*0.95], axis=0)
        return all_boxes, all_scores

    def get_rois(self):
        '''
        1)get roi from feature map
        2)roi align or roi pooling. Here is roi align
        :return:
        all_level_rois: [N, 7, 7, C]
        all_level_proposals : [N, 5]
        all_level_proposals is matched with all_level_rois

        因为产生rois的时候打乱了self.first_stage_decode_boxes的顺序， 而到时候解码的时候应该让rois和正确的reference box对应，
        所以要重新产生一个匹配的all_level_proposals
        '''
        levels = self.assign_level()

        all_level_roi_list = []
        all_level_proposal_rotate_list = []
        all_level_proposal_horizontal_list = []

        with tf.variable_scope('crop_roi_and_roi_align'):
            for i in range(self.min_level, self.max_level + 1):
                level_i_proposal_indices = tf.reshape(tf.where(tf.equal(levels, i)), [-1])
                level_i_rotate_proposals = tf.gather(self.rpn_proposals_boxes, level_i_proposal_indices)

                level_i_rotate_proposals = tf.cond(
                    tf.equal(tf.shape(level_i_rotate_proposals)[0], 0),
                    lambda: tf.constant([[0, 0, 1, 1, -90]], dtype=tf.float32),
                    lambda: level_i_rotate_proposals
                )  # to avoid level_i_proposals batch-size is 0, or this project will be broken when BP gradients

                all_level_proposal_rotate_list.append(level_i_rotate_proposals)

                level_i_horizon_proposals = get_horizon_minAreaRectangle(level_i_rotate_proposals, False)
                level_i_horizon_proposals = clip_boxes_to_img_boundaries(level_i_horizon_proposals,
                                                                         img_shape=self.img_shape)

                xmin, ymin, xmax, ymax = tf.unstack(level_i_horizon_proposals, axis=1)

                h = tf.maximum(ymax-ymin, 0)
                w = tf.maximum(xmax-xmin, 0)
                x_c = (xmax+xmin) // 2
                y_c = (ymax+ymin) // 2
                theta = tf.ones_like(h) * -90
                level_i_horizontal_proposals = tf.transpose(tf.stack([x_c, y_c, h, w, theta]))
                all_level_proposal_horizontal_list.append(level_i_horizontal_proposals)

                img_h, img_w = tf.cast(self.img_shape[1], tf.float32), tf.cast(self.img_shape[2], tf.float32)
                normalize_ymin = ymin / img_h
                normalize_xmin = xmin / img_w
                normalize_ymax = ymax / img_h
                normalize_xmax = xmax / img_w

                level_i_cropped_rois = tf.image.crop_and_resize(self.feature_pyramid['P%d' % i],
                                                                boxes=tf.transpose(tf.stack([normalize_ymin, normalize_xmin,
                                                                                             normalize_ymax, normalize_xmax])),
                                                                box_ind=tf.zeros(shape=[tf.shape(level_i_rotate_proposals)[0], ],
                                                                                 dtype=tf.int32),
                                                                crop_size=[self.roi_size, self.roi_size],
                                                                name='CROP_AND_RESIZE'
                                                                )
                if cfgs.USE_MASK:
                    '''
                    RRPN, affine rotation. We implement it with rotated mask.
                    '''
                    roi_mask = tf_wrapper.get_mask_tf(level_i_rotate_proposals, self.roi_size)  # [<300, 14, 14]

                    roi_mask = tf.stack([roi_mask for _ in range(256)], axis=3)
                    level_i_cropped_rois = level_i_cropped_rois * roi_mask

                level_i_rois = slim.max_pool2d(level_i_cropped_rois,
                                               [self.roi_pool_kernel_size, self.roi_pool_kernel_size],
                                               stride=self.roi_pool_kernel_size)
                all_level_roi_list.append(level_i_rois)

            all_level_rois = tf.concat(all_level_roi_list, axis=0)
            all_level_rotate_proposals = tf.concat(all_level_proposal_rotate_list, axis=0)
            all_level_horizontal_proposals = tf.concat(all_level_proposal_horizontal_list, axis=0)

            return all_level_rois, all_level_rotate_proposals, all_level_horizontal_proposals

    def fast_rcnn_net(self):

        with tf.variable_scope('fast_rcnn_net'):
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                flatten_rois_features = slim.flatten(self.fast_rcnn_all_level_rois)

                net = slim.fully_connected(flatten_rois_features, 1024, scope='fc_1')
                if self.use_dropout:
                    net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout')

                net = slim.fully_connected(net, 1024, scope='fc_2')

                fast_rcnn_scores = slim.fully_connected(net, self.num_classes + 1, activation_fn=None,
                                                        scope='classifier')

                fast_rcnn_encode_boxes = slim.fully_connected(net, self.num_classes * 5, activation_fn=None,
                                                              scope='regressor')

                return fast_rcnn_encode_boxes, fast_rcnn_scores

    def fast_rcnn_find_positive_negative_samples(self, reference_boxes):
        '''
        when training, we should know each reference box's label and gtbox,
        in second stage
        iou >= 0.5 is object
        iou < 0.5 is background
        :param reference_boxes: [num_of_input_boxes, 5]
        :return:
        reference_boxes_mattached_gtboxes: each reference box mattched gtbox, shape: [num_of_input_boxes, 5]
        object_mask: indicate box(a row) weather is a object, 1 is object, 0 is background
        category_label: indicate box's class, one hot encoding. shape: [num_of_input_boxes, num_classes+1]
        '''

        with tf.variable_scope('fast_rcnn_find_positive_negative_samples'):
            gtboxes = tf.cast(
                tf.reshape(self.gtboxes_and_label[:, :-1], [-1, 5]), tf.float32)  # [M, 5]

            ious = tf_wrapper.get_iou_matrix_tf(reference_boxes, gtboxes, use_gpu=cfgs.IOU_USE_GPU, gpu_id=0)
            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)  # [N, ]
            reference_boxes_mattached_gtboxes = tf.gather(gtboxes, matchs)  # [N, 5]
            max_iou_each_row = tf.reduce_max(ious, axis=1)
            # [N, ]
            if self.use_angle_condition:
                cond1 = tf.greater_equal(max_iou_each_row, self.fast_rcnn_positives_iou_threshold)

                # angle condition
                gtboxes_angles = reference_boxes_mattached_gtboxes[:, -1]  # tf.unstack(anchors_matched_gtboxes, axis=1)
                reference_boxes_angles = reference_boxes[:, -1]  # tf.unstack(anchors, axis=1)

                cond2 = tf.less_equal(tf.abs(gtboxes_angles - reference_boxes_angles), self.boxes_angle_threshold)

                positives = tf.cast(tf.logical_and(cond1, cond2), tf.int32)
            else:
                positives = tf.cast(tf.greater_equal(max_iou_each_row, self.fast_rcnn_positives_iou_threshold),
                                    tf.int32)

            object_mask = tf.cast(positives, tf.float32)  # [N, ]
            # when box is background, not caculate gradient, so give a weight 0 to avoid caculate gradient

            label = tf.gather(self.gtboxes_and_label[:, -1], matchs)  # [N, ]
            label = tf.cast(label, tf.int32) * positives  # background is 0
            # label = tf.one_hot(category_label, depth=self.num_classes + 1)

            return reference_boxes_mattached_gtboxes, object_mask, label

    def fast_rcnn_minibatch(self, reference_boxes):
        with tf.variable_scope('fast_rcnn_minibatch'):

            reference_boxes_mattached_gtboxes, object_mask, label = \
                self.fast_rcnn_find_positive_negative_samples(reference_boxes)

            positive_indices = tf.reshape(tf.where(tf.not_equal(object_mask, 0.)), [-1])

            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                          tf.cast(self.fast_rcnn_minibatch_size*self.fast_rcnn_positives_ratio, tf.int32))

            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices, begin=[0], size=[num_of_positives])

            negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.)), [-1])
            num_of_negatives = tf.minimum(tf.shape(negative_indices)[0],
                                          self.fast_rcnn_minibatch_size - num_of_positives)

            negative_indices = tf.random_shuffle(negative_indices)
            negative_indices = tf.slice(negative_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_reference_boxes_mattached_gtboxes = tf.gather(reference_boxes_mattached_gtboxes,
                                                                    minibatch_indices)
            object_mask = tf.gather(object_mask, minibatch_indices)
            label = tf.gather(label, minibatch_indices)
            label_one_hot = tf.one_hot(label, self.num_classes + 1)

            return minibatch_indices, minibatch_reference_boxes_mattached_gtboxes, object_mask, label_one_hot

    def fast_rcnn_loss(self):
        with tf.variable_scope('fast_rcnn_loss'):
            minibatch_indices, minibatch_reference_boxes_mattached_gtboxes, minibatch_object_mask, \
            minibatch_label_one_hot = self.fast_rcnn_minibatch(self.fast_rcnn_all_level_rotate_proposals) #######################

            # minibatch_reference_boxes = tf.gather(self.fast_rcnn_all_level_horizontal_proposals, minibatch_indices)
            minibatch_reference_boxes = tf.gather(self.fast_rcnn_all_level_rotate_proposals, minibatch_indices)

            minibatch_encode_boxes = tf.gather(self.fast_rcnn_encode_boxes,
                                               minibatch_indices)  # [minibatch_size, num_classes*5]

            minibatch_scores = tf.gather(self.fast_rcnn_scores, minibatch_indices)

            positive_proposals_in_img = draw_box_with_color(self.img_batch,
                                                            minibatch_reference_boxes * tf.expand_dims(
                                                                   minibatch_object_mask, 1),
                                                            text=tf.shape(tf.where(tf.equal(minibatch_object_mask, 1.0)))[0])

            negative_mask = tf.cast(tf.logical_not(tf.cast(minibatch_object_mask, tf.bool)), tf.float32)
            negative_proposals_in_img = draw_box_with_color(self.img_batch,
                                                            minibatch_reference_boxes * tf.expand_dims(negative_mask, 1),
                                                            text=tf.shape(tf.where(tf.equal(minibatch_object_mask, 0.0)))[0])

            tf.summary.image('/positive_proposals', positive_proposals_in_img)
            tf.summary.image('/negative_proposals', negative_proposals_in_img)

            minibatch_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=minibatch_encode_boxes,
                                                                    reference_boxes=minibatch_reference_boxes,
                                                                    scale_factors=self.scale_factors)

            minibatch_softmax_scores = tf.gather(slim.softmax(self.fast_rcnn_scores), minibatch_indices)
            top_k_scores, top_k_indices = tf.nn.top_k(minibatch_softmax_scores[:, 1], k=5)

            top_detections_in_img = draw_boxes_with_scores(self.img_batch,
                                                           boxes=tf.gather(minibatch_decode_boxes, top_k_indices),
                                                           scores=top_k_scores)
            tf.summary.image('/top_5', top_detections_in_img)


            # encode gtboxes
            minibatch_encode_gtboxes = \
                encode_and_decode.encode_boxes(
                    unencode_boxes=minibatch_reference_boxes_mattached_gtboxes,
                    reference_boxes=minibatch_reference_boxes,
                    scale_factors=self.scale_factors)

            # [minibatch_size, num_classes*5]
            minibatch_encode_gtboxes = tf.tile(minibatch_encode_gtboxes, [1, self.num_classes])

            class_weights_list = []
            category_list = tf.unstack(minibatch_label_one_hot, axis=1)
            for i in range(1, self.num_classes+1):
                tmp_class_weights = tf.ones(shape=[tf.shape(minibatch_encode_boxes)[0], 5], dtype=tf.float32)
                tmp_class_weights = tmp_class_weights * tf.expand_dims(category_list[i], axis=1)
                class_weights_list.append(tmp_class_weights)
            class_weights = tf.concat(class_weights_list, axis=1)  # [minibatch_size, num_classes*5]

            # loss
            with tf.variable_scope('fast_rcnn_classification_loss'):
                fast_rcnn_classification_loss = slim.losses.softmax_cross_entropy(logits=minibatch_scores,
                                                                                  onehot_labels=minibatch_label_one_hot)
                # if DEBUG:
                #     print_tensors(minibatch_scores, 'minibatch_scores')
                #     print_tensors(classification_loss, '2nd_cls_loss')
            with tf.variable_scope('fast_rcnn_location_loss'):
                fast_rcnn_location_loss = losses.l1_smooth_losses(predict_boxes=minibatch_encode_boxes,
                                                                  gtboxes=minibatch_encode_gtboxes,
                                                                  object_weights=minibatch_object_mask,
                                                                  classes_weights=class_weights)
                slim.losses.add_loss(fast_rcnn_location_loss)

            return fast_rcnn_location_loss, fast_rcnn_classification_loss

    def fast_rcnn_proposals(self, decode_boxes, scores):
        '''
        mutilclass NMS
        :param decode_boxes: [N, num_classes*5]
        :param scores: [N, num_classes+1]
        :return:
        detection_boxes : [-1, 5]
        scores : [-1, ]

        '''

        with tf.variable_scope('fast_rcnn_proposals'):
            category = tf.argmax(scores, axis=1)

            object_mask = tf.cast(tf.not_equal(category, 0), tf.float32)

            decode_boxes = decode_boxes * tf.expand_dims(object_mask, axis=1)  # make background box is [0 0 0 0, 0]
            scores = scores * tf.expand_dims(object_mask, axis=1)

            decode_boxes = tf.reshape(decode_boxes, [-1, self.num_classes, 5])  # [N, num_classes, 5]

            decode_boxes_list = tf.unstack(decode_boxes, axis=1)
            score_list = tf.unstack(scores[:, 1:], axis=1)
            after_nms_boxes = []
            after_nms_scores = []
            category_list = []
            for per_class_decode_boxes, per_class_scores in zip(decode_boxes_list, score_list):

                if self.top_k_nms:
                    top_k_scores, top_k_indices = tf.nn.top_k(per_class_scores, k=self.top_k_nms)
                    per_class_scores = top_k_scores
                    per_class_decode_boxes = tf.gather(per_class_decode_boxes, top_k_indices)

                valid_indices = tf_wrapper.nms_rotate_tf(boxes_list=per_class_decode_boxes,
                                                         scores=per_class_scores,
                                                         iou_threshold=self.fast_rcnn_nms_iou_threshold,
                                                         max_output_size=self.fast_rcnn_nms_max_boxes_per_class,
                                                         use_gpu=cfgs.NMS_USE_GPU)

                after_nms_boxes.append(tf.gather(per_class_decode_boxes, valid_indices))
                after_nms_scores.append(tf.gather(per_class_scores, valid_indices))
                tmp_category = tf.gather(category, valid_indices)

                category_list.append(tmp_category)

            all_nms_boxes = tf.concat(after_nms_boxes, axis=0)
            all_nms_scores = tf.concat(after_nms_scores, axis=0)
            all_category = tf.concat(category_list, axis=0)

            # all_nms_boxes = boxes_utils.clip_boxes_to_img_boundaries(all_nms_boxes,
            #                                                          img_shape=self.img_shape)

            scores_large_than_threshold_indices = \
                tf.reshape(tf.where(tf.greater(all_nms_scores, self.show_detections_score_threshold)), [-1])

            all_nms_boxes = tf.gather(all_nms_boxes, scores_large_than_threshold_indices)
            all_nms_scores = tf.gather(all_nms_scores, scores_large_than_threshold_indices)
            all_category = tf.gather(all_category, scores_large_than_threshold_indices)

            # top_k_scores, top_k_indices = tf.nn.top_k(all_nms_scores, k=self.second_maximum_boxes_per_img)
            # return tf.gather(all_nms_boxes, top_k_indices), top_k_scores)
            return all_nms_boxes, all_nms_scores, tf.shape(all_nms_boxes)[0], all_category  # num of objects

    def fast_rcnn_predict(self):

        with tf.variable_scope('fast_rcnn_predict'):
            fast_rcnn_softmax_scores = slim.softmax(self.fast_rcnn_scores)  # [-1, num_classes+1]

            fast_rcnn_encode_boxes = tf.reshape(self.fast_rcnn_encode_boxes, [-1, 5])

            # reference_boxes = tf.tile(self.fast_rcnn_all_level_horizontal_proposals, [1, self.num_classes])
            reference_boxes = tf.tile(self.fast_rcnn_all_level_rotate_proposals, [1, self.num_classes]) # [N, 5*num_classes]

            reference_boxes = tf.reshape(reference_boxes, [-1, 5])   # [N*num_classes, 5]
            fast_rcnn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=fast_rcnn_encode_boxes,
                                                                    reference_boxes=reference_boxes,
                                                                    scale_factors=self.scale_factors)

            # mutilclass NMS
            fast_rcnn_decode_boxes = tf.reshape(fast_rcnn_decode_boxes, [-1, self.num_classes*5])
            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                self.fast_rcnn_proposals(fast_rcnn_decode_boxes, scores=fast_rcnn_softmax_scores)

            return fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category












