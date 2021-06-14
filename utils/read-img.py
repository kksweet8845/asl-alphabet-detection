import tensorflow as tf
import os
import glob
from utils import to_tfRecord, parse_tfRecords
import cv2
import numpy as np
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_dir", help="The path of input directory")
    parser.add_argument("-o", "--output_file", help="The name of output file")

    args = parser.parse_args()


    if args.input_dir:
        root_path = args.input_dir
    else:
        root_path = "./data/train"
    
    if args.output_file:
        tfRecord_file = args.output_file
    else:
        tfRecord_file = "images.tfrecords"

    to_tfRecord(root_path, type="jpg", tf_output_file=tfRecord_file)

    parsed_records = parse_tfRecords(tfRecords_file=tfRecord_file)

    with open("preview.log", 'w') as file:
        for record in parsed_records:
            file.write("="*60)
            file.write("\n")
            file.write(f"Label : {record['label']}\n")
            file.write(f"Height: {record['height']}\n")
            file.write(f"Width: {record['width']}\n")
        

    

