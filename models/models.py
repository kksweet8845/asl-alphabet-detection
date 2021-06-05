import tensorflow as tf


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model



class AlexNet(Model):
    def __init__(self, classes=5):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()

        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(classes)

    def call(self, x, trainin=True):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)



