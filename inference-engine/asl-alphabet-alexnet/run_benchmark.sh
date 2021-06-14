#!/bin/bash

# activate env
source /opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer/venv_tf2/bin/activate

# OPENVINO_PATH=/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
# SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python benchmark.py --model_src /home/nober/repo/ai-chip/asl-alphabet/openvino-models/alexNet_model_src_optimized \
            --model_name AlexNet --input_dir /home/nober/repo/ai-chip/asl-alphabet/inference-engine/asl-alphabet-alexnet/test