# -*- coding: utf-8 -*-
#Lesson 3 - Load Data From CSV by using Pandas, CSV, Numpy

url = "https://goo.gl/vhm1eU"
'''0, 91, 80, 0, 0, 32.4, 0.601, 27, 0'''

# Load CSV using Pandas from URL
import pandas

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
print(data.shape)
''' (768, 9)  768 entree de 9 elements'''

# Load CSV using csv from URL
import csv
import urllib2 # have to use urllib2

response = urllib2.urlopen(url)
cr = csv.reader(response)
''' cr = csvreader object rempli avec les données '''

# Load CSV using numpy from URL
import numpy as np
import urllib

raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
print dataset.shape