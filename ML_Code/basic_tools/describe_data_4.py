# -*- coding: utf-8 -*-
#Lesson 4 - Understand Data with Descriptive Statistics
# DataFrame.describe / DataFrame.head / DataFrame.shape /
# DataFrame.dtypes / DataFrames.corr


# DataFrame.describe with Pandas
# Ne fonctionne qu'a partir d'un read from csv
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html
import pandas

url = "https://goo.gl/vhm1eU"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
description = data.describe()
print(description)
'''
             pedi
count  768.000000  # nombre d'entrées dans le csv
mean    33.240885  # retourne la moyenne des données
std     11.760232  # retourne la différence moyenne entre les valeurs
min     21.000000  # This method returns the minimum of the value in the csv.
25%     24.000000  # This method return the element at the 25% position
50%     29.000000  # This method return the element at the 50% position
75%     41.000000  # This method return the element at the 75% position
max     81.000000  # This method returns the maximum of the value in the csv
'''

# DataFrame.head with Pandas
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html?highlight=head#pandas.DataFrame.head

first_rows = data.head(5)
print(first_rows)
''' Print the n first_rows of the csv '''

# DataFrame.shape with Pandas
print(data.shape)
''' (nb_entries, nb_columns) '''

# DataFrame.dtypes with Pandas
print(data.dtypes)
''' return the type of variables '''

# DataFrame.corr with Pandas
# DataFrame.corr = covariance(A,B)/std(A)*std(B)
'''Compute pairwise correlation of columns, excluding NA/null values'''
print(data.corr)

df = pandas.DataFrame({'A': range(4), 'B': [2*i for i in range(4)]})
#covariance(A,B) = E((Ai - mean(A) * (Bi - mean(B))) / n-1
#so we have to iterate on each value of A and B
#get the product and then divide by n - 1                   
print('df[A].corr(df[B])' ,df['A'].corr(df['B']))

''''df.loc[2, 'B'] = 6
print(df['A'].corr(df['B']))'''