# Doudou Khallil

# Larger CNN digit_recognition from MNIST DATASET
# On ajoute un couple ConvLayer,Pooling
# On ajoute un Fully connected layer

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_pixels = X_train.shape[1] * X_train.shape[2]
# reshape to be [n_samples][channels][width][height]
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype( 'float32' )
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype( 'float32')

# on normalize entre 0 et 1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode output
# si y = 4
# [0, 0, 0, 1, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    # 784-> 32C2-> P2-> 15C2-> P2-> Drop0.2-> Flat-> 128-> 50 -> 10
    model = Sequential()
    model.add(Conv2D(30,(5,5), input_shape=(1,28,28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(15,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.2))

    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))    
    
    model.add(Dense(num_classes, activation='softmax'))        
    model.compile(loss='categorical_crossentropy',optimizer='adam',
        metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train,y_train,validation_data=(X_test, y_test), epochs=10,
    batch_size=200,verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error : %.2f%%" % (100-scores[1]*100))