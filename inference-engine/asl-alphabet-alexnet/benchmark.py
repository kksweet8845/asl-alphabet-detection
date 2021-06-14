import cv2
import numpy as np
from openvino.inference_engine import IECore
import logging as log
import sys
from argparse import ArgumentParser
import glob, os
import time



def build_argparser():

    parser = ArgumentParser(add_help=False)

    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit')


    args.add_argument('-s', '--model_src', help='Required. Path to an .xml or .onnx file with a trained model.',
                        required=True, type=str)
    args.add_argument("-m", '--model_name', help="required. The name of <model_name>.xml",
                        required=True, type=str)
    args.add_argument("-i", "--input_dir", type=str, default="", required=True,
                        help="Required, The directory contains the test image")



    args.add_argument('-l', '--cpu_extension',
                        help="Optional. Required for CPU custom layers",
                        type=str)
    args.add_argument('-d', '--device',
                        help="Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable.",
                        default="CPU", type=str)
    args.add_argument('--labels', help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument('-nt', '--number_top', help='Optional. Number of top results', default=10, type=int)

    return parser

def main():
    log.basicConfig(format="[%(levelname)s] %(message)s", level=log.INFO, stream=sys.stdout)

    args = build_argparser().parse_args()

    log.info("Create Inference Engine")

    ie = IECore()
    # ie.add_extension(args.cpu_extension, "CPU")

    root_dir = args.model_src
    model_name = args.model_name
    # model = "/home/nober/repo/ai-chip/asl-alphabet/openvino-models/alexnet/AlexNet.xml"
    model = f"{root_dir}/{model_name}.xml"
    weights = f"{root_dir}/{model_name}.bin"
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model, weights=weights)
    device = args.device
    number_top = args.number_top

    log.info(f"Input info : {net.input_info}")

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Read and pre-process input images
    data_dir = args.input_dir
    print(data_dir)

    n, c, h, w = net.input_info[input_blob].input_data.shape
    # images = np.ndarray(shape=(n, c, h, w))

    # for fitpath in glob.glob(os.path.join(f"{data_dir}", "*.jpg"):

    filepath_iter = iter(glob.glob(os.path.join(f"{data_dir}", "*.jpg")))
    result_ls = []

    duration = 0
    iteration = 0
    while True:
        images = np.ndarray(shape=(n, c, h, w))
        try:
            for i in range(n):
                filepath = next(filepath_iter)
                print(filepath)
                iteration += 1
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                image = np.expand_dims(image, axis=2)
                if image.shape[:-1] != (h, w):
                    log.warning(f"Image {args.input} is resized from {image.shape[:-1]} to {(h, w)}")
                    image = cv2.resize(image, (h, w))
                image = image.transpose((2, 0, 1))
                images[i] = image
        except StopIteration:
            break

        log.info("loading model to the plugin")
        exec_net = ie.load_network(network=net, device_name=device)

        log.info("Starting inference in synchronous mode")
        start = time.process_time_ns()
        res = exec_net.infer(inputs={input_blob: images})
        end = time.process_time_ns()

        duration += (end - start)

        log.info("Processing output blob")
        res = res[out_blob]


        result_ls.append((filepath.split('/')[-1], res))



    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None



    classid_str = "classid"
    probability_str = "Probability"

    print("Result\n")
    print("="*60)

    for filename, res in result_ls:
        print(f"Image {filename}")
        for i, probs in enumerate(res):
            probs = np.squeeze(probs)
            top_ind = np.argsort(probs)
            top_ind = np.flip(top_ind)
            print(classid_str, probability_str)
            print(f"{'-'* len(classid_str)} {'-' * len(probability_str)}")
            for id in top_ind:
                det_label = f"{id}"
                label_length = len(det_label)
                space_num_before = (len(classid_str) - label_length) // 2
                space_num_after = len(classid_str) - (space_num_before + label_length) + 2
                print(f"{' ' * space_num_before}{det_label}"
                    f"{' ' * space_num_after}{probs[id]:.7f}")

            print("\n")

    print("#"*60)

    print(f"Benchmark analysis\n",
          f"Count      : {iteration}  iterations\n",
          f"Duration   : {duration}   nsecs\n",
          f"Throughput : {1e9/duration} fps\n")

if __name__ == "__main__":


    main()






