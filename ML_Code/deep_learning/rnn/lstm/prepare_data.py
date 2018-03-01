
# We will use how to prepare all king of data for LSTM

from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from math import sqrt

# I. Prepare Numerical Data
# II. Prepare Categorical Data
# III. Prepare Sequence with Varied L
# IV. Supervised Sequence

#I.1 Normalization
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
series = Series(data)
values = series.values
values = values.reshape((len(values),1))
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(values)
print( 'Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
normalized = scaler.transform(values)
print(normalized)
inversed = scaler.inverse_transform(normalized)

#I.2 Standardization
data = [1.0, 5.5, 9.0, 2.6, 8.8, 3.0, 4.1, 7.9, 6.3]
series = Series(data)
values = series.values
values = values.reshape((len(values),1))
scaler = StandardScaler()
scaler = scaler.fit(values)
print( 'Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
standardized = scaler.transform(values)
print standardized

#II. Integer Encoding + One hot encoding

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold' , 'cold' , 'warm' , 'cold ' , 'hot' , 'hot' , 'warm' , 'cold' , 'warm' , 'hot' ]
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

#III.1 Sequence Padding
from keras.preprocessing.sequence import pad_sequences
sequences = [[1,2,3,4],
             [1,2,3],
             [1],]
# by default add '0' from the 'start'
padded = pad_sequences(sequences) # padding='post's-> end
print padded

#III.2 Sequence Truncation
# remove x from the array if too big, or add 0, both from start
truncated=pad_sequences(sequences,maxlen=2)#truncating='post'->end
print(truncated)

#IV. Pandas shift
from pandas import DataFrame
df = DataFrame()
df['t'] = [x for x in range(10)]
print(df)
df['t-1'] = df['t'].shift(1) # 2 1
#df['t+1'] = df['t'].shift(-1) # 2 3


