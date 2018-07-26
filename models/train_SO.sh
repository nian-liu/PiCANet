#!/usr/bin/env sh
VGG_MODEL_DIR=/home/nianliu/Deeplearning_codes/caffe/caffe-master/models/VGG_ILSVRC_16_layers/vgg16_20M.caffemodel
RES50_MODEL_DIR=/home/nianliu/Deeplearning_codes/caffe/caffe-master/models/ResNet/ResNet-50-model.caffemodel
CAFFE_DIR=../caffe
export GLOG_log_dir=./log

$CAFFE_DIR/build/tools/caffe train --solver=solver.prototxt --weights $VGG_MODEL_DIR
#$CAFFE_DIR/build/tools/caffe train --solver=solver.prototxt --weights $RES50_MODEL_DIR