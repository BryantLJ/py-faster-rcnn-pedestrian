#!/bin/bash

LOG="output/faster_rcnn_end2end/inria/`date +'%Y-%m-%d-%H-%M-%S'.log`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/lj_train_net.py
