#-*- coding: utf-8 -*-

# SMALL REGRESSION MODEL

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# on defini les inputs
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
# on defini les outputs des inputs
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

# retourne l'erreur de la prediction entre y_true et y_pred
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

# avec l'erreur on modifie les [bias et weights] pour chaque input
optimizer = tf.train.GradientDescentOptimizer(0.01)
# on utilise l'optimiseur Gradient Descent en passant l'erreur en paramètre
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
sess.run(init)
for i in range(300):
    _, loss_value = sess.run((train,loss))
    print(loss_value)

print(sess.run(y_pred))
