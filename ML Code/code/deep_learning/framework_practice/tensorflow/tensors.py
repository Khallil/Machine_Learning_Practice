#-*- coding: utf-8 -*-

import tensorflow as tf
# Les Tensors

# I - Rank  11
# II - Shape 25
# III - Datatype 49
# IV - Eval Tensors 54
# V - Printing Tensors 63

# I - Rank
v = tf.Variable(23,tf.int16) # R0 = un scalaire x
l = tf.Variable([23,12,8],tf.int16) # R1 = une liste x1,xn
t = tf.Variable([[7],[11]],tf.int16) # R2 = un tableau xn,yn
# I.1 Higher Ranks on défini pas la valeurs seulement la shape
c = tf.zeros([1,1,1]) # R3 = un tableau xn,yn,zn
hc = tf.zeros([1,2,3,4]) # R4 = un tableau xn,yn,zn,tn
r = tf.rank(c)
s = tf.Session()
print s.run(r)

# to get get the scalar value it's simple
lv = l[1] # for R1 tensor

# II - Shape
print v.shape #() pour R0
print l.shape #(3,) pour R1, 3 item dans la liste
print t.shape #(2,1) pour R2, 2 row, 1 item par row
print c.shape #(1,1,1) pour R3, 1 row, 1 profondeur, 1 item par row

# Change the Shape
#   On peut changer la shape, par contre il faut respecter le n d'elements
#   obtenu par le produits des sizes des shape
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
#yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!

# III - DataType

# On peut caster un tensor avec tf.cast
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)

# IV - Eval Tensors
# SA SERT A QUOI ? Je sais pas c'est comme s.run(tensor)

constant = tf.constant([1,2,3])
tensor = constant * constant
print tensor.eval(session=s)
print s.run(tensor)

# V - Printing Tensors
# L'interet c'est que quand on run, un tensor, le tensor
# appelle tout ce dont il a besoin, sauf que si il a apelle
# tf.Print (pour récupérer la valeur) et bien tf.print va afficher
# la valeur retourné dans le terminal 

t = tf.constant(3) #tf.int32 par défaut
tf.Print(t,[t])
t = tf.Print(t,[t]) # [3] au runtime
result = t + 1
print s.run(result) #4