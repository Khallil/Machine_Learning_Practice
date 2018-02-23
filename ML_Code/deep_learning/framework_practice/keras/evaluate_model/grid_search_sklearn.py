
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
import time

start = time.time()

def create_model(optimizer='rmsprop',init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation= 'relu' ))
    model.add(Dense(8, activation= 'relu' ))
    model.add(Dense(1, activation= 'sigmoid' ))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("../pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

model = KerasClassifier(build_fn=create_model, verbose=0)
optimizers = ['rsmprop','adam']
inits =['normal','uniform']
epochs = [1,2]
batches = [5,10]
param_grid = dict(optimizer=optimizers,epochs=epochs,batch_size=batches,init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
print X.shape
print Y.shape
grid_result = grid.fit(X,Y)

print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
#for mean,stdev,param in zip(means, stds,params):
 #   print("%f (%f) with : %r" % (mean,stdev,params))
end = time.time()
print(end - start)