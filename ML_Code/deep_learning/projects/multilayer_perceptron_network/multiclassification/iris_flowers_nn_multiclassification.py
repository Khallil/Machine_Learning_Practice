#Khallil Doudou
#-*- coding: utf-8 -*-

# On encode la colonne Y our la multiple classification
# On set la loss fonction loss='categorical_crossentropy'
# On set l'actiation fonction de l'output model.add(Dense(3, activation='softmax'))

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

#  4 inputs -> [8 hidden nodes] -> 3 outputs
def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(8, input_dim=4,activation='relu'))
    model.add(Dense(3, activation='softmax'))
    #compile model
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)

dataframe = read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]
print Y

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y


# fit le model
#estimator = KerasClassifier(build_fn=baseline_model,epochs=200,batch_size=5, verbose=0)

#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))