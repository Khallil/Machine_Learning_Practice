# -*- coding: utf-8 -*-
#Lesson 2 - Get Around In Python, NumPy, Matplotlib and Pandas.

import numpy
import pandas
#                           n        n+1 
myarray = numpy.array([[1, 2, 3], [-1, 5, 6]]) 
rownames = ['n', 'n+1']
colnames = ['one', 'two', 'three']

# creating a pandas dataframe
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

'''
   one  two  three
a    1    2      3
b    -1    5      6
'''
#creating a plot with matplotlib

import matplotlib.pyplot as plt

plt.plot(myarray)
plt.ylabel('myarray')
plt.show()