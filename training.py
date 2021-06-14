
from utils import Trainer
from models import AlexNet
from utils import to_tfRecord, freeze_graph
import argparse
import tensorflow as tf


if __name__ == "__main__":

    # Training data
    train_tfd = "./tfrecords/train.tfrecords"
    train_input = "./processed_data/train"
     # Transform the dataset to tfrecords format
    ts = to_tfRecord(train_input, type="jpg", tf_output_file=train_tfd)

    # Testing data
    test_tfd = "./tfrecords/test.tfrecords"
    test_input = "./processed_data/test"
     # Transform the dataset to tfrecords format
    tts = to_tfRecord(test_input, type="jpg", tf_output_file=test_tfd)

    # Construct the Trainer to train the model.
    inputs = tf.keras.Input(shape=(50, 50, 1), dtype="float16")
    model = AlexNet(inputs, classes=5)
    print(model.inputs)
    EPOCH = 50
    Trainer(train_tfd, test_tfd, model,  EPOCH)

    # Save the model

    # tf.saved_model.save(model, "./model_src/")
    tf.keras.models.save_model(
        model,
        "./alexNet_model_src",
        overwrite=True,
        include_optimizer=True,
        save_format=None)

    # freeze_graph(model, './model_src', 'frozen_graph')
