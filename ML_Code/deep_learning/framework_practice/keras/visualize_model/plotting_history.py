
# Visualize evolution of model with plot

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy',
             optimizer='adam', metrics=['accuracy'])

numpy.random.seed(7)

dataset = numpy.loadtxt("../pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

history = model.fit(X,Y,validation_split=0.33,epochs=300, batch_size=10, verbose=0)

# summarize history for accuracy
print(history.history.keys())

# acc during training
plt.plot(history.history[ 'acc' ])
# acc during testing
plt.plot(history.history[ 'val_acc' ])
plt.title( 'model accuracy' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'upper left' )
plt.show()

# loss during training
plt.plot(history.history[ 'loss' ])
# loss during testing
plt.plot(history.history[ 'val_loss' ])
plt.title( 'model loss' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend([ 'train' , 'test' ], loc= 'upper left' )
plt.show()