# LSTM Variable Input to One
# On entraine le network avec des taille differentes
# pour qu'il comprenne de facon generique l'alphabet

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

numpy.random.seed(7)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# pour learning
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# pour predict
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# on cree les colonnes
num_inputs = 1000
max_len = 5 #max len of input
dataX = []
dataY = []
for i in range(num_inputs):
    start = numpy.random.randint(len(alphabet)-2)
    end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
    seq_in = alphabet[start:end+1]
    seq_out = alphabet[end + 1]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print seq_in,"->",seq_out

X = pad_sequences(dataX,maxlen=max_len, dtype='float32')
X = numpy.reshape(X, (X.shape[0], max_len,1))
X = X / float(len(alphabet))
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(32,input_shape=(X.shape[1], 1)))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=500,batch_size=1,verbose=2)

scores = model.evaluate(X,y,verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

for i in range(20):
    pattern_index = numpy.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
    x = numpy.reshape(x,(1,max_len, 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", result)

