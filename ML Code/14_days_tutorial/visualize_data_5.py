# -*- coding: utf-8 -*-
#Lesson 5 - Understand Data with Visualization

import matplotlib.pyplot as plt
import pandas

# Scatter Plot Matrix ith Diabete dataset
#affiche un point pour positionner le rapport entre deux
#valeurs de 2 colonnes à la meme n position
from pandas.tools.plotting import scatter_matrix

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
scatter(data)

# Scatter Plot Matrix with custom dataset
#data_custom = pandas.read_csv('df_csv')
#print data_custom
'''scatter_matrix(data_custom)'''

# DataFrame hist
# Draw histogram of the DataFrame’s series using matplotlib / pylab.
'''data.hist()'''
# ça affiche un graphique pour représenter la quantité
# de chaque valeur pour mieux visualiser la répartition
# des valeurs

# DataFrame boxplot
'''data_custom.boxplot()'''

# DataFrame plot
# Draw graphics of different types
# df.plot(kind='')
'''
    ‘bar’ or ‘barh’ for bar plots (x=n_element, y=valeur de n_element) /
         barh = affichage horizontal
    ‘hist’ for histogram (better use data.hist()) 
    ‘box’ for boxplot (better use data.boxplot())
         (donne une idée de la plage de valeurs et affiche la médiane en vert)
    ‘kde’ or 'density' for density plots
         (donne la fréquence d'apparition des valeurs sous forme de courbes)
    ‘area’ for area plots (comme 'bar' mais sous forme de zone progressive ! add (,stacked=False))
    ‘scatter’ for scatter plots (comme scatter_matrix mais en plus chiant à utiliser)
    ‘hexbin’ for hexagonal bin plots (comme scatter mais assemble les points en hexadecimaux)
    ‘pie’ for pie plots (un camembert qui marche pas bien ! add(,subplots=True))
'''
''' data_custom.plot(kind=''); '''
plt.show()

