
# Save model with JSON and H5

# Model architecture is saved in JSON file
# Weights of NN is saved in H5 file
# Then we load the JSON
# Then we load the H5 in the loaded_JSON

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import numpy

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy',
             optimizer='adam', metrics=['accuracy'])

numpy.random.seed(7)

dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model.fit(X,Y,epochs=150, batch_size=10, verbose=0)
scores = model.evaluate(X,Y,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# SERIALIZE MODEL TO JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialoize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later ...

# load json
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# create model from json
loaded_model = model_from_json(loaded_model_json)
# add weights in the model
loaded_model.load_weights("model.h5")
print "Loaded model from disk"

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
yaml_file = open( 'model.yaml' , 'r' )
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


loaded_model.compile(loss='binary_crossentropy',
             optimizer='rmsprop', metrics=['accuracy'])
scores = loaded_model.evaluate(X,Y,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))
