
from utils import Trainer
from models import AlexNet
from utils import to_tfRecord
import argparse


if __name__ == "__main__":

    # Training data
    train_tfd = "train.tfrecords"
    train_input = "./data/train"
     # Transform the dataset to tfrecords format
    to_tfRecord(train_input, type="jpg", tf_output_file=train_tfd)

    # Testing data
    test_tfd = "test.tfrecords"
    test_input = "./data/test"
     # Transform the dataset to tfrecords format
    to_tfRecord(test_input, type="jpg", tf_output_file=test_tfd)

    # Construct the Trainer to train the model.
    model = AlexNet()
    EPOCH = 30
    Trainer(train_tfd, test_tfd, model,  EPOCh)




