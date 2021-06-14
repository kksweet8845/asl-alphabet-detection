import cv2
import numpy as np
from openvino.inference_engine import IECore
import logging as log
import sys
from argparse import ArgumentParser


def build_argparser():

    parser = ArgumentParser(add_help=False)

    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit')
    args.add_argument('-m', '--model', help='Required. Path to an .xml or .onnx file with a trained model.',
                        required=True, type=str)
    args.add_argument('-i', '--input', help='Required. Path to an image file',
                        required=True, type=str)
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

    root_dir = "/home/nober/repo/ai-chip/asl-alphabet/openvino-models/alexNet_model_src_optimized"
    model_name = "AlexNet"
    # model = "/home/nober/repo/ai-chip/asl-alphabet/openvino-models/alexnet/AlexNet.xml"
    model = f"{root_dir}/{model_name}.xml"
    weights = f"{root_dir}/{model_name}.bin"
    log.info(f"Loading network:\n\t{model}")
    net = ie.read_network(model=model, weights=weights)

    log.info(f"Input info : {net.input_info}")

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    print(f"Samples {n}")
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        print(image.shape)
        image = np.expand_dims(image, axis=2)
        if image.shape[:-1] != (h, w):
            log.warning(f"Image {args.input} is resized from {image.shape[:-1]} to {(h, w)}")
            image = cv2.resize(image, (h,w))
        # image = image.transpose((2, 0, 1))
        images[i] = image

    # Loading model to plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info(f"Top {args.number_top} results: ")
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None

    classid_str = "classid"
    probability_str = "Probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)
        print(f"Image {args.input}")
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

if __name__ == "__main__":


    main()

