#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.caltech import caltech
from datasets.inria import inria
import datasets.imdb
import caffe
import numpy as np
import sys

if __name__ == '__main__':


    #set up global variable 'cfg' for train
    train_solver = sys.path[4] + '/' + 'models/pascal_voc_person/VGG16/solver.prototxt'
    pretrained_caffemodel = sys.path[4] + '/' + 'data/imagenet_models/VGG16.caffemodel'
    max_iterations = 4000
    train_imdb = inria(train = 1)
    roidb = get_training_roidb(train_imdb)
    
    # set up global varibles for validation
    validation_network = sys.path[4]+'/'+ 'models/pascal_voc_person/VGG16/test.prototxt'
    validation_imdb = inria(test = 1)

    # set up global caffe mode
    cfg_file = sys.path[4] + '/' + 'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    
    if 1:
        train_solver = 'models/pascal_voc_person/VGG16/solver.prototxt'
        pretrained_caffemodel = 'data/imagenet_models/VGG16.caffemodel'
        validation_network = 'models/pascal_voc_person/VGG16/test.prototxt'
        cfg_file = 'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    
    cfg_from_file(cfg_file)
    
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # setup output result directory
    output_dir = get_output_dir(train_imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # train net
    train_net(train_solver, roidb, output_dir,
              pretrained_model=pretrained_caffemodel,
              max_iters=max_iterations,validation_cfg = (validation_network,validation_imdb))

