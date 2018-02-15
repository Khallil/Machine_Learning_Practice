#-*- coding: utf-8 -*-

import time   
import tensorflow as tf

start = time.time()

# un TENSOR est une variable, qui débute au scalaire jusqu'au matrices,
# 3D tensor, n-Tensor

# création des VARIABLES
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# création de MATRICES
c = tf.constant([[1.0, 2.0], [3.0,4.0]])
d = tf.constant([[1.0, 1.0], [0.0,1.0]])

# crétation de FONCTION
# création de l'expression symbolique
add = tf.add(a,b)
mul = tf.matmul(c,d)

# création de la SESSION
sess = tf.Session()

# création de lassociation de valeur BINDING
binding = {a: 1.5, b: 2.5}

# on GET LE RETOUR DE LA SESSION
somme = sess.run(add, feed_dict=binding)
produit = sess.run(mul)

print somme 
print produit

print a.dtype
print c.dtype

end = time.time()
print(end - start)