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
    model.fit(train_x,train_y,verbose=0,epochs=epoch,batch_size=1)
    return model.evaluate(test_x,test_y,verbose=0)[1]

# On paramètre tout ici :)
initializers = ['glorot_normal','glorot_uniform','random_uniform','random_normal']
#activation = ['...','',]
#final activation = ['...','',]
epochs = [1,2,3]
n_layers = [1,2,3]
n_cells = [10,30,60]

n_repeats = 3
scores = DataFrame()
for epoch in epochs:
    print 'epoch : %d'%(epoch)
    for init in initializers:
        print ' init : %s'%(init)
        for n_l in n_layers:
            print '  n_layers : %d'%(n_l)
            #for n_c in n_cells:
            start_time = time.time()
            perf_v = list()
            for i in range(n_repeats):
                perf = fit_model(train_x,train_y,test_x,test_y,n_class,epoch,init,n_l,10)
                perf_v.append(perf)
            scores[str(epoch),str(n_layers),init] = perf_v
            print '   n_cells : %d - %f'%(10,time.time() - start_time)

print(scores.describe())
print scores.mean().sort_values(ascending=False)

perf = 'accu' #/loss/others metrics
scores.boxplot()
pyplot.ylabel(perf)
pyplot.xlabel('epochs')
pyplot.show()

