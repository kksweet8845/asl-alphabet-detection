
#!/bin/bash

# activate
pyenv3.7;

OPENVINO_PATH=/opt/intel/openvino_2021.3.394/deployment_tools/model_optimizer
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


while getopts "i:o:n:?" argv
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
        ?)
            INPUTMODEL=""
            exit
            ;;
    esac
done


python $OPENVINO_PATH/mo.py --framework tf --input_model $SCRIPTDIR/../$INPUTMODEL --output_dir $OUTPUTDIR --model_name $MODELNAME


