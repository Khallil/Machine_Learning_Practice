# -*- coding: utf-8 -*-

# Jason Brownlee forked
# First Neural Network

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
for item in dataset:
    print item
# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

#DEFINE MODEL
model = Sequential()
#Fully connected layers are defined using the Dense class.
# We can specify the number of neurons in the layer as the first argument 
# and specify the activation function using the activation argument

#The first hidden layer has 12 neurons and expects
#8 input variables (e.g. input dim=8). 
model.add(Dense(12, input_dim=8, activation= 'relu' ))

#The second hidden layer has 8 neurons  
model.add(Dense(8, activation= 'relu'))

#Finally the output layer has 1 neuron to predict the class (onset of diabetes or not).
#We use a sigmoid activation function on the output
#layer to ensure our network output is between 0 and 1
model.add(Dense(1, activation= 'sigmoid' ))

#COMPILE MODEL
#logarithmic loss, which for a
#binary classification problem is defined in Keras as binary crossentropy.
model.compile(loss= 'binary_crossentropy' ,
#use the
#efficient gradient descent algorithm adam
 optimizer= 'adam' , 
#because it is a classification problem, we will collect and report the
#classification accuracy as the metric.
 metrics=[ 'accuracy' ])

#FIT MODEL 
# We can also set the number of instances that are evaluated before
# a weight update in the network is performed called the batch size
model.fit(X, Y, epochs=150, batch_size=10)

#EVALUATE MODEL
#on passe le train dataset dans l'evaluation
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))