#!/bin/bash

# activate
pyenv3.8

OPENVINO_PATH=/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"




python $OPENVINO_PATH/mo.py --framework caffe --input_model ~/openvino_models/public/mtcnn/mtcnn-o/mtcnn-o.caffemodel --input_proto ~/openvino_models/public/mtcnn/mtcnn-o/mtcnn-o.prototxt --output_dir ~/openvino_models/saved_model/mtcnn-o 


python $OPENVINO_PATH/mo.py --framework caffe --input_model ~/openvino_models/public/mtcnn/mtcnn-p/mtcnn-p.caffemodel --input_proto ~/openvino_models/public/mtcnn/mtcnn-p/mtcnn-p.prototxt --output_dir ~/openvino_models/saved_model/mtcnn-p 