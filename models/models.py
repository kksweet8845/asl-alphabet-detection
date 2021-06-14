import tensorflow as tf


from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
from tensorflow.nn import softmax
from tensorflow.math import sigmoid


# model = keras.Sequential([
#   layers.Conv2D(32, 3, input_shape=(50, 50, 1), activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(1024, activation='relu'),
#   layers.Dense(128, activation='relu'),
#   layers.Dropout(0.5),
#   layers.Dense(num_classes, activation=None),
#   layers.Softmax()
# ])


class AlexNet(Model):
    def __init__(self, inputs, classes=5):
        super(AlexNet, self).__init__()
        self.inputs = inputs
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.maxp1 = MaxPool2D()
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.maxp2 = MaxPool2D()
        self.flatten = Flatten()

        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.drop = Dropout(0.5)
        self.d3 = Dense(classes, activation=None)
        # self.soft = softmax()

    def call(self, x, trainin=True):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.drop(x)
        x = self.d3(x)
        x = softmax(x)
        # x = sigmoid(x)
        return x



