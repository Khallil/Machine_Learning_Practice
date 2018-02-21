#-*- coding: utf-8 -*-

import tensorflow as tf
# Graph et Session

# I - Graph - Name Operations 11
# II - Graph - Using Specified Device 51
# III - Graph - Tensor-like Objects
# IV - Session

# I  - Graph - Name Operations
with tf.name_scope("outer"):
  c_2 = tf.constant(2, name="c")  # => operation named "outer/c"

# II - Graph - Using Specified Device
with tf.device("/device:CPU:0"):
  # Operations created in this context will be pinned to the CPU.
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # Operations created in this context will be pinned to the GPU.
  result = tf.matmul(weights, img)

# III - Graph - Tensor-like Objects
tf.convert_to_tensor() # peut convertir ces objets en tensors :
#tf.Variable
#numpy.ndarray
#list (and lists of tensor-like objects)
#Scalar Python types: bool, float, int, str

# IV - Session - Run tricks
# Ci dessous la particularité c'est que y va être apellé 2 fois par le code
# mais va être calculé seulement une fois puis réutilisé
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer
with tf.Session() as sess:
  # Run the initializer on `w`.
  sess.run(init_op)

  # Evaluate `output`. `sess.run(output)` will return a NumPy array containing
  # the result of the computation.
  print(sess.run(output))

  # Evaluate `y` and `output`. Note that `y` will only be computed once, and its
  # result used both to return `y_val` and as an input to the `tf.nn.softmax()`
  # op. Both `y_val` and `output_val` will be NumPy arrays.
  y_val, output_val = sess.run([y, output]

# IV.1 - Runtrick 2
# A noter que le feeding d'un tf.placehoder est unique au run
# ce n'est pas une initialisation comme avec une variable

# Define a placeholder that expects a vector of three floating-point values,
# and a computation that depends on it.
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # Feeding a value changes the result that is returned when you evaluate `y`.
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # Raises `tf.errors.InvalidArgumentError`, because you must feed a value for
  # a `tf.placeholder()` when evaluating a tensor that depends on it.
  sess.run(y)

  # Raises `ValueError`, because the shape of `37.0` does not match the shape
  # of placeholder `x`.
  sess.run(y, {x: 37.0})

  
