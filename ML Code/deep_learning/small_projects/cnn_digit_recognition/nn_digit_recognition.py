# Doudou Khallil

# NN digit_recognition from MNIST DATASET

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
# flatten 28*28 images to a 784 vector for each image
# [x0,x1,...,x783]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype( 'float32' )
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype( 'float32')

# on normalize entre 0 et 1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode output
# si y = 4
# [0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print y_train
print y_test
print num_classes


def baseline_model():
    # 784 -> 784 -> 10
    model = Sequential()
                      # 784                 784
    model.add(Dense(num_pixels, input_dim=num_pixels,
        kernel_initializer='normal',activation='relu'))
                        #10
    model.add(Dense(num_classes, kernel_initializer='normal',
        activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',
        metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=10,
    batch_size=200,verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error : %.2f%%" % (100-scores[1]*100))