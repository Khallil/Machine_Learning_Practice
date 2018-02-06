# -*- coding: utf-8 -*-
#Lesson 6 - Prepare For Modeling by Pre-Processing Data

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Binarizer


import pandas
import numpy
url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

# separate array into input and output components
# ?
X = [[0, 0],[1, 1], [2, 2], [3, 3], [4, 4],[5, 5]]
#X = array[:,0:8] #recupère les colonnes de 0 à 8 exclu
Y = array[:,8]   #recupère la colonne 8

#numpy.set_printoptions(precision=1) #affiche seulement 3 décimales
print("-- Basic Data Form --")

#print(X[0:5,:]) 
#print(Y)
#Custom Scaler
#    scaler.mean_ = 0
#    scaler.scale_ = 1

scaler = StandardScaler().fit(X) 
minmaxscaler = MinMaxScaler().fit(X)
maxabsscaler = MaxAbsScaler().fit(X)
robustscaler = RobustScaler().fit(X)
quantilescaler = QuantileTransformer().fit(X)
binarizer = Binarizer().fit(X)

# summarize transformed data
print("-- StandarScaler.transform --")
rescaledX = scaler.transform(X)     
print(rescaledX[0:5,:]) #affiche seulement les 5 premmières lignes
print("-- Normalizer.transform(l2) --")
X_normalized = normalize(X, norm='l2')
print(X_normalized[0:5,:])
print("-- MinMaxScaler.transform() --")
X_minmaxscalerized = minmaxscaler.transform(X)
print(X_minmaxscalerized[0:5,:])
print("-- MaxAbsScaler.transform() --")
X_maxabsscalerized = maxabsscaler.transform(X)
print(X_maxabsscalerized[0:5,:])
print("-- RobustScaler.transform() --")
X_robustscalerized = robustscaler.transform(X)
print(X_robustscalerized[0:5,:])
print("-- QuantileScaler.transform() --")
X_quantilescalerized = quantilescaler.transform(X)
print(X_quantilescalerized[0:5,:])
print("-- Binarizer.transform() --")
X_binarized = binarizer.transform(X)
print(X_binarized[0:5,:])

'''
#data = [[0, 0], [0, 0], [1, 3], [3, 3]]
data = [[0, 0], [3, 3]]
#

#---------------------

#STANDARD SCALER

scaler = StandardScaler().fit(data) # l'objet stock le mean(moyenne) et le std(standard deviation)
print(scaler.mean_)
print(scaler.scale_) 
    #mean = µ = 1
    #std = square( ( (0 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (3 - 1)^2 / N)
    #                1    + 1 +  0 + 4 = 6/4 = 1.5
    #                square(1.5)  = 1.22474487
#------------------

#NORMALIZER #Normalize each row to unit norm

X_normalized = normalize(data, norm='l2')
print(X_normalized) # 3  - 1.5 = 1.5 --> 1.5 / 3


#------------------

#TRANSFORM

print(scaler.transform(data)) # Perform standardization by centering and scaling
     #0 - 1 = - 1 | soustrais la valeur par scaler.mean_
     #-1 / 1.22474487 = -0.81649658185 | divise la valeur par scaler.scale_ (std)
 
print(scaler.transform([[2, 2]]))
     #on peut ensuite passer nimporte quelles valeurs au scaler
'''