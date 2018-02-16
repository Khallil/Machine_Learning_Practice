#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# I - GRAPH - 17
# II - TENSORBOARD - 36
# III - SESSION - 41
# IV - DATASET - 54
# V - LAYERS - 70
# VI - FEATURE COLUMN - 88

# I - GRAPH
# A computational GRAPH composed of Operations and Tensors

# 'a' est un TENSOR, 'tf.constant' est l' OPERATION qui instancie le tensor
a = tf.constant(3.0, dtype=tf.float32)

# pareil pour 'b'
b = tf.constant(4.0) # also tf.float32 implicitly

# total est un tensor, '+' est l'opération qui additione des tensors entre eux
total = a + b

# Le print ici va simplement afficher les propriétés du tensor
print(a) #Tensor("Const:0", shape=(), dtype=float32)
print(b) #Tensor("Const_1:0", shape=(), dtype=float32)
print(total) #Tensor("add:0", shape=(), dtype=float32)
# A noter que : Tensors are named after the operation that produces them
# followed by an output index

# II - TENSORBOARD
# Outil permettant d'afficher le graph créé
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

# III - SESSION
# La session c'est le runtime, c'est à dire qu'il va lancer les opérations du graph
sess = tf.Session()
# sess.run va lancé l'opération du tensor et retourner l'output
print(sess.run(total)) #7.0
# on peut utiliser sess.run plusieurs fois pour relancer l'opération

# tf.constant retourne un tensor avec une valeure fixe
# pour assigner une valeure dynamique on utilise tf.placeholder
x = tf.placeholder(tf.float32) 
# et feed_dict dans le sess.run
print(sess.run(x,feed_dict={x:2})) #2.0

# IV - DATASET
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,]]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

# V - LAYERS
# Graph part -- on crée une matrice avec shape[x,y],
# ici on dit qu'on a N row de dimension 3
x = tf.placeholder(tf.float32, shape=[None, 3])
# crée un layer qui prends des inputs en paramètre et produit 1 seul output
linear_model = tf.layers.Dense(units=1)
# y est également à la liste des outputs
y = linear_model(x)
# cette opération permet l'init des variable dans le layer (les weights)
# TIPS : A créer a la fin car elle init seulement les variable créé au dessus d'elle
init = tf.global_variables_initializer()

# Session part -- on lance l'init des valeurs du layer
sess.run(init)
# on lance le layer
print(sess.run(y, {x: [[1,2,3],
                       [4,5,6]]}))

# VI - FEATURES COLUMNS
# Ce sont des colonnes mais sous formes de features columns tensorflow
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}
# numpy form
[5,'sports']
[10,'sports']
[8,'gardening']
[9,'gardening']

# Création de l'output layer de catégorisation
# tf.feature_column.categorical_column_with_vocabulary_list for string
department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])
# on crée une dense column with .indicator_column
department_column = tf.feature_column.indicator_column(department_column)

# tf.feature_column.numeric_column for integer, qui produit une dense column
columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

# On crée l'input layer avec les denses columns
inputs = tf.feature_column.input_layer(features, columns)

# On init les variables dans l'input layer
var_init = tf.global_variables_initializer()
# On init les tables pour la categorical column
table_init = tf.tables_initializer()
# on run
sess = tf.Session()
sess.run((var_init,table_init))
# lancer tf.feature_column.input_layer, crée un tableau avec les x et le y sous
# la forme "one-hot" pour l'output
print(sess.run(inputs))
'''[[  1.   0.   5.]
    [  1.   0.  10.]
    [  0.   1.   8.]
    [  0.   1.   9.]]'''


