#-*- coding: utf-8 -*-

import tensorflow as tf

# création des float
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# création de l'expression symbolique
add = tf.add(a,b)

# création de la session
sess = tf.Session()

# création de lassociation de valeur (binding)
binding = {a: 1.5, b: 2.5}

# lancement de la session
c = sess.run(add, feed_dict=binding)

print(c)