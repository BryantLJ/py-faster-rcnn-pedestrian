#!/usr/bin/env python
'''
this is used to change detection results **.pkl to *.txt format, so that we can use MATLAB toolbox for evaluating the results.
The results is on Caltech-Usa-test dataset
2016/11/23. by BryantLJ
'''

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
import cPickle
from reportlab.platypus.para import lengthSequence
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil

imdb = inria(test=1)

with open('/home/bryant/py-faster-rcnn/output/faster_rcnn_end2end/inria/vgg16_faster_rcnn_iter_1000/detections.pkl','rb') as f:
    detect_boxes = cPickle.load(f)
    
result = detect_boxes[1]

## change pkl format to .txt, then use MATLAB toolbox 
for i in range(len(result)):
    image_path = imdb.image_path_at(i)
    abs_path = '/home/bryant/MATLAB-tools/Matlab evaluation_labeling code3.2.1/data-INRIA/res/FasterRCNN/'
    txt_path = image_path[-21:-11]+'.txt'
    txt_path = txt_path.replace('v','V')
    txt_path = abs_path + txt_path 
    if os.path.exists(txt_path):
        os.remove(txt_path)
#shutil.rmtree('/home/bryant/MATLAB-tools/Matlab evaluation_labeling code3.2.1/results/')
## change pkl format to .txt, then use MATLAB toolbox 
for i in range(len(result)):
    image_path = imdb.image_path_at(i)
    res = result[i]
    im = cv2.imread(imdb.image_path_at(i))
    #vis_detections(im,'person',res)
    abs_path = '/home/bryant/MATLAB-tools/Matlab evaluation_labeling code3.2.1/data-INRIA/res/FasterRCNN/'
    txt_path = image_path[-21:-11]+'.txt'
    txt_path = txt_path.replace('v','V')
    txt_path = abs_path + txt_path 
    fram_id = int(image_path[-9:-4])+1
    with open(txt_path,'a') as f:
        for box in res:
            box_res = '{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(fram_id,box[0],box[1],box[2]-box[0]+1,box[3]-box[1]+1,box[4])
            #if fram_id!=0:
            f.write(box_res)
    f.close()

imdb._do_python_eval(result)
exit()