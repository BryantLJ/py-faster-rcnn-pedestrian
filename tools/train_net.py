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
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_person import pascal_voc_person
from datasets.caltech import caltech
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

if __name__ == '__main__':

    #set up global variable 'cfg' for train
    train_solver = sys.path[4] + '/' + 'models/pascal_voc_person/VGG16/solver.prototxt'
    cfg_file = sys.path[4] + '/' + 'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    pretrained_caffemodel = sys.path[4] + '/' + 'data/imagenet_models/VGG16.caffemodel'
    
    #===========================================================================
    # train_solver = 'models/pascal_voc_person/VGG16/solver.prototxt'
    # cfg_file = 'models/pascal_voc_person/VGG16/vgg16_faster_rcnn.yml'
    # pretrained_caffemodel = 'data/imagenet_models/VGG16.caffemodel'
    #===========================================================================
    
    print train_solver
    print cfg_file
    print pretrained_caffemodel
    max_iterations = 45000

    cfg_from_file(cfg_file)
    cfg.GPU_ID = 0

    # set up global caffe mode
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # set up train imdb
    #imdb = pascal_voc_person('person_trainval','2007')
    #imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    
    imdb = caltech(train = 1)
    
    roidb = get_training_roidb(imdb)
    print '{:d} roidb entries'.format(len(roidb))
    
    # setup output result directory
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    # train net
    train_net(train_solver, roidb, output_dir,
              pretrained_model=pretrained_caffemodel,
              max_iters=max_iterations)

