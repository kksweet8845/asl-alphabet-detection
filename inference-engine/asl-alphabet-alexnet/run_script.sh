#!/bin/bash

# activate env
source /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/venv_tf2/bin/activate

# OPENVINO_PATH=/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
# SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python main.py -m /home/nober/repo/ai-chip/asl-alphabet/openvino-models/alexnet/AlexNet.xml -i ./a_test.jpg