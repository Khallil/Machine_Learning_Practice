# Generate Text from save model based on prediction

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "wonderland.txt"
raw_text = open(filename).read()

raw_text = raw_text.lower()
chars = sorted(list(set(raw_text)))
chars_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters : ", n_chars)
print("Total Vocab : ", n_vocab)

seq_length = 100
dataX=[]
dataY=[]
#prepare the dataset
for i in range(0,n_chars-seq_length,1):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns", n_patterns)

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
filename = "weights-improvement-19-1.9435-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')

# generate characters
# prends une phrase pour commencer
start = numpy.random.randint(0,len(dataX)-1)
pattern = dataX[start]
print "Seed:"
# reconverti les chiffres en characters pour l'affichage
print "\"",''.join([int_to_char[value] for value in pattern]),"\"")

for i in range(1000):
    x = numpy.reshape(pattern, (1,len(pattern),1))
    x = x/float(n_vocab)
    prediction = model.predict(x,verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print "\nDone."
