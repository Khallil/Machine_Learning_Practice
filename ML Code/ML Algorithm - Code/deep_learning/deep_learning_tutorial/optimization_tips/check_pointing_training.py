# CheckPoint training

# We save weights in a unique hdf5 file, and update this file each time
# an improvement of accuracy is done during the training

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model

# !!! If file already exist !!!
# We get the best weights we get before 
#model.load_weights("weights.best.hdf5")

model.compile(loss='binary_crossentropy',
             optimizer='adam', metrics=['accuracy'])

numpy.random.seed(7)
filepath="weights.best.hdf5"

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
checkpoint = ModelCheckpoint(filepath, monitor= 'val_acc' , verbose=0, save_best_only=True,
mode='max') 
callbacks_list = [checkpoint]
model.fit(X,Y, validation_split=0.33,epochs=150,batch_size=10,callbacks=callbacks_list,verbose=0)

scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))