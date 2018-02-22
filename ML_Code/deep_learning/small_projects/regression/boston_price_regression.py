# Regression
# On change loss fonction : model.compile(loss='mean_squared_error')
# # KerasRegressor pour la regression

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input and output variables
X = dataset[:,0:13]
Y = dataset[:,13]

def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal',activation='relu'))
    model.add(Dense(6, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

seed = 7
numpy.random.seed(seed)

estimators = []
estimators.append(( 'standardize' , StandardScaler()))
estimators.append(( 'mlp' , KerasRegressor(build_fn=baseline_model, epochs=150, batch_size=10,
verbose=0)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))