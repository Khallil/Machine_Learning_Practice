# --coding: utf-8 -- #

from numpy import array
from random import seed, uniform,shuffle
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
from pandas import DataFrame
import time

# crée un dataset
seed(1)
n_class = 6 # nombre des classes
n_sample = 10 # nombre de sample per class
dataset = list()
for i in range(n_class):
    n = ((n_class*n_sample)/n_sample)*(i+1)
    for x in range(n_sample):
        dataset.append([uniform(n-n_class,n),i])

# on mélange et on split, train/test, x/y
shuffle(dataset)
split = 0.8
l_train = int(len(dataset)*0.8)
train = dataset[0:l_train]
test = dataset[l_train:]
train_x,train_y = array([x[0] for x in train]),array([x[1] for x in train])
test_x,test_y = array([x[0] for x in test]),array([x[1] for x in test])
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

def fit_model(train_x,train_y,test_x,test_y,n_class,epoch,init,
                n_layers,n_cells):
    model = Sequential()
    model.add(Dense(n_cells,input_dim=1,kernel_initializer=init,activation='relu'))
    for n in range(n_layers-1):
        model.add(Dense(n_cells, kernel_initializer=init,activation='relu'))
    model.add(Dense(n_class,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    start_time = time.time()
    model.fit(train_x,train_y,verbose=1,epochs=epoch,batch_size=1)
    print(time.time() - start_time)
    loss = model.evaluate(test_x,test_y,verbose=0)[1]
    del model
    return loss

epoch = 400
init = 'glorot_uniform'
n_l = 3
n_c = 40

fit_model(train_x, train_y, test_x, test_y, n_class, epoch, init, n_l, n_c)
