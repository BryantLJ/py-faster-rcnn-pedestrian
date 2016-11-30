#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.pascal_voc_person import pascal_voc_person
from datasets.caltech import caltech
from datasets.inria import inria
import caffe
import argparse
import pprint
import time, os, sys

if __name__ == '__main__':
    gpu_id = 0
    prototxt = 'models/pascal_voc_person/VGG16/test.prototxt'
    caffemodel = 'output/faster_rcnn_end2end/inria/vgg16_faster_rcnn_iter_3000.caffemodel'
    cfg_file = sys.path[4] + '/' +'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    
    prototxt = sys.path[4] + '/' + prototxt
    caffemodel = sys.path[4] + '/' + caffemodel
    
    comp_mode = False
    visable = True
    max_image = 10000

    cfg_from_file(cfg_file)
    cfg.GPU_ID = gpu_id

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    
    imdb = inria(test = 1)
    test_net(net, imdb, max_per_image=max_image, vis=visable)
