from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os,sys
sys.path.insert(0, '../../')

from libs.networks.slim_nets import resnet_v1
from libs.networks.slim_nets import mobilenet_v1
from libs.networks.slim_nets import inception_resnet_v2
from libs.networks.slim_nets import vgg

from libs.configs import cfgs

def get_network_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       spatial_squeeze=True):
    if net_name not in ['resnet_v1_50', 'mobilenet_224', 'inception_resnet', 'vgg16', 'resnet_v1_101']:
        raise ValueError('''not include network: {}, net_name must in [resnet_v1_50, mobilenet_224, 
                            inception_resnet, vgg16, resnet_v1_101]
                         '''.format(net_name))

    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfgs.WEIGHT_DECAY[net_name])):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        num_classes=num_classes,
                                                        is_training=is_training,
                                                        global_pool=global_pool,
                                                        output_stride=output_stride,
                                                        spatial_squeeze=spatial_squeeze
                                                        )

        return logits, end_points
    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfgs.WEIGHT_DECAY[net_name])):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         spatial_squeeze=spatial_squeeze
                                                         )
        return logits, end_points

