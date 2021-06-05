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
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.d1(x)
        print(x.shape)
        return self.d2(x)



