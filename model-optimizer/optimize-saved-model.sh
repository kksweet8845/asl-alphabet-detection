
#!/bin/bash

# activate
pyenv3.8;

OPENVINO_PATH=/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


while getopts "i:o:n:s:?" argv
do
    case $argv in
        i)
            INPUTMODEL=$OPTARG
            ;;
        o)
            OUTPUTDIR=$OPTARG
            ;;
        n)
            MODELNAME=$OPTARG
            ;;
        s)
            SAVED_MODEL_DIR=$OPTARG
            ;;
        ?)
            INPUTMODEL=""
            ;;
    esac
done

echo $MODELNAME
echo $SAVED_MODEL_DIR
echo $OUTPUTDIR

python $OPENVINO_PATH/mo.py --framework tf -b 1 --output_dir ./alexNet_model_src_optimized --model_name AlexNet --saved_model_dir /home/nober/repo/ai-chip/asl-alphabet/alexNet_model_src


