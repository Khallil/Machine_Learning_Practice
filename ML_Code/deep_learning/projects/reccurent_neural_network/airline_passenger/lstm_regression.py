# Time series prediction with LSTM NN - Regression

# l37 : On peut augmenter le look_back
# l41 : On peut modifer la shape en input du lstm : time_step = look_back
# l49 : On peut conserver la cell state du lstm durant tous le training 
# l53 : On peut ajouter un autre LSTM layer, with return=True au layer d'avant

from pandas import read_csv
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python',
    skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),0] # 0 pour garder les item sur la meme feature
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX),numpy.array(dataY)

look_back = 3
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)
# on change la shape de 95,1 a 95,1,1
# on peut change la time step par le nombre de colonne en inversant les 2 :
trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
#trainX = numpy.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = numpy.reshape(testX,(testX.shape[0],testX.shape[1],1))
#testX = numpy.reshape(testX,(testX.shape[0],1,testX.shape[1]))

batch_size = 1
model = Sequential()
# on set stateful a True pour empecher le reset de la memoire

model.add(LSTM(4, batch_input_shape=(batch_size,look_back,1), stateful=True,
return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size,look_back,1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

for i in range(100):
    model.fit(trainX,trainY,epochs=1,batch_size=batch_size,verbose=2,shuffle=False)
    model.reset_states()

trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: (%.2f RMSE)' % trainScore)
#    math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))))
print('Test Score: (%.2f RMSE)' % (
    math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))))

# shift predictions for plotting
# empty_like, cree une nouvelle array de la meme forme et meme type de variable
# mais avec des valeurs randoms
trainPredictPlot = numpy.empty_like(dataset)
# numpy nan set toutes les valeurs de l'array avec nan
trainPredictPlot[:, :] = numpy.nan
# de 1 a 94  = remplir avec les valeurs de trainPredict
# on decale le remplissage de l'array de la taille de look_back pour l'affichage du plot
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
 
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
# de 97 a 143 = remplit avec testPredict
# pareil ici aussi on decale
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset),:] = testPredict
#testPredictPlot[len(trainPredict):len(dataset),:] = testPredict

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()