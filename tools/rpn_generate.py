#!/usr/bin/env python

# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from datasets.caltech import caltech
from rpn.generate import imdb_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, sys

if __name__ == '__main__':

    gpu_id = 0
    prototxt = 'models/pascal_voc_person/VGG16/test.prototxt'
    caffemodel = 'output/faster_rcnn_end2end/caltech/stride_7_iter_20000_model/vgg16_faster_rcnn_iter_20000.caffemodel'
    cfg_file = sys.path[4] + '/' +'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    
    prototxt = sys.path[4] + '/' + prototxt
    caffemodel = sys.path[4] + '/' + caffemodel


    cfg_from_file(cfg_file)
    cfg.GPU_ID = gpu_id

    # RPN test settings
    cfg.TEST.RPN_PRE_NMS_TOP_N = -1
    cfg.TEST.RPN_POST_NMS_TOP_N = 400


    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]

    imdb = caltech(test= 1)
    imdb_boxes = imdb_proposals(net, imdb)
