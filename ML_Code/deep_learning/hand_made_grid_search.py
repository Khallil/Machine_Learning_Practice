# --coding: utf-8 -- #

from numpy import array
from random import seed, uniform,shuffle
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot
from pandas import DataFrame

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

def fit_model(train_x,train_y,test_x,test_y,n_class,epoch,init):
    model = Sequential()
    model.add(Dense(10,input_dim=1,kernel_initializer=init))
    model.add(Dense(n_class,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_x,train_y,verbose=0,epochs=epoch,batch_size=1)
    return model.evaluate(test_x,test_y,verbose=0)[1]

initializers = ['glorot_normal','glorot_uniform','random_uniform','random_normal']
epochs = [1,2,3] 
n_repeats = 5
scores = DataFrame()
for epoch in epochs:
    print 'epoch : %d'%(epoch)
    for init in initializers:
        print 'init : %s'%(init)
        perf_v = list()
        for i in range(n_repeats):
            perf = fit_model(train_x,train_y,test_x,test_y,n_class,epoch,init)
            perf_v.append(perf)
        scores[str(epoch),init] = perf_v

print(scores.describe())

perf = 'accu' #/loss/others metrics
scores.boxplot()
pyplot.ylabel(perf)
pyplot.xlabel('epochs')
pyplot.show()

