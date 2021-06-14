from keras.applications.mobilenet_v2 import MobileNetV2
from read_img import read_img_from_directory
import keras

# l_map = {
#     'A' : 0,
#     'B' : 1,
#     'C' : 2,
#     'D' : 3,
#     'E' : 4,
#     'F'
# }

l_map = {chr(key): key-65 for key in range(65, 91)}
print(l_map)

x_train, y_train = read_img_from_directory('./data2/train', 10, l_map)
x_test, y_test = read_img_from_directory('./data2/test', 10, l_map)

x_train = x_train / 255.0
x_test = x_test / 255.0

model = MobileNetV2(input_shape=(200, 200, 1), include_top=True, weights=None, classes=10)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
model.fit(x=x_train, y=y_train, batch_size=200, shuffle=True, epochs=20, validation_split=0.2)

model.save('MobileNetV2.h5')



