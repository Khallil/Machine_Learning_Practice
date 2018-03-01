# Time series prediction with Multilayer Perceptron NN - Regression

# si on augmente la fenetre du temps
# il faut augmenter la taille du reseau egalement
# deeper and larger with more epochs too


from pandas import read_csv
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense

dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python',
    skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [],[]
    # pourquoi il enleve les 2 dernieres valeurs ? (390 et 432)
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back),0] # 0 pour garder les item sur la meme feature
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX),numpy.array(dataY)
#plt.plot(dataset)
#plt.show(),,d,d,fyoutube
look_back = 3
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)

model = Sequential()
model.add(Dense(14, input_dim = look_back,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,epochs=400,batch_size=2,verbose=0)
trainScore = model.evaluate(trainX,trainY,verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
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

plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()