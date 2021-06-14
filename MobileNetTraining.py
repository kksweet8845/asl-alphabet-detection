
from utils import Trainer
from models import AlexNet
from utils import to_tfRecord, freeze_graph
import argparse
import tensorflow as tf
import os


if __name__ == "__main__":

    # Training data
    train_tfd = "./tfrecords/train.tfrecords"
    train_input = "./data2/train"
     # Transform the dataset to tfrecords format
    ts = to_tfRecord(train_input, type="jpg", tf_output_file=train_tfd)

    # Testing data
    test_tfd = "./tfrecords/test.tfrecords"
    test_input = "./data2/test"
     # Transform the dataset to tfrecords format
    tts = to_tfRecord(test_input, type="jpg", tf_output_file=test_tfd)

    # Construct the Trainer to train the model.

    inputs = tf.keras.Input(shape=(200, 200, 1), dtype="float16")
    # model = AlexNet(inputs, classes=5)
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(200,200,1), include_top=True, weights=None, classes=26)
    print(model.inputs)
    EPOCH = 30
    Trainer(train_tfd, test_tfd, model,  EPOCH, ts=ts, tts=tts)

    # Save the model

    # tf.saved_model.save(model, "./model_src/")
    os.makedirs("./mobilenet_model_src", exist_ok=True)
    tf.keras.models.save_model(
        model,
        "./mobilenet_model_src",
        overwrite=True,
        include_optimizer=True,
        save_format=None)

    # freeze_graph(model, './model_src', 'frozen_graph')
