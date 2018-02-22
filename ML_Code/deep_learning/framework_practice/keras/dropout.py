
# EXAMPLE OF USE OF DROPOUT 
# we use the dropout for each hidden layers
# we use maxnorm to limit he max value of the weights
# we use SGD to otpimize the weights

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 60 input | 60 neuron in hidden layer | 1 output
def create_baseline():
    #create model
    model = Sequential()
    model.add(Dense(60, input_dim=60,kernel_initializer='normal',activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal' , activation= 'relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
    #compile model
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = read_csv("/home/doudou/Documents/IA/ML Code/ML Algorithm - Code/deep_learning/small_projects/binaryclassification/sonar.csv", header=None)
dataset = dataframe.values
# split into input and output variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# evaluate model with standardized dataset
estimators = []
estimators.append(( 'standardize' , StandardScaler()))
estimators.append(( 'fit' , KerasClassifier(build_fn=create_baseline, epochs=200, batch_size=5,
verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))