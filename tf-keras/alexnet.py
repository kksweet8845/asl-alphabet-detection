from keras import layers
from read_img import read_img_from_directory
import keras

l_map = {
    'A' : 0,
    'B' : 1,
    'C' : 2,
    'D' : 3
}
x_train, y_train = read_img_from_directory('./data/train', 4, l_map)
x_test, y_test = read_img_from_directory('./data/test', 4, l_map)

x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = 4


model = keras.Sequential([
  layers.Conv2D(32, 3, input_shape=(200, 200, 1), activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(1024, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation=None),
  layers.Softmax()
])



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
model.fit(x=x_train, y=y_train, batch_size=200, shuffle=True, epochs=10, validation_split=0.2)

model.summary()
model.save('AlexNet.h5')