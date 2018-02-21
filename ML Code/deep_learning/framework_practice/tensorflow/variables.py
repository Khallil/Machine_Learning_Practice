#-*- coding: utf-8 -*-

import tensorflow as tf
# Les Variables

# I - Create Variable 11
# II - Init Variables 51
# III - Using Variables 67
# IV - Sharing Variables 77

# I - Create Variable
# Best way is to use get_variable
# which create a 3D tensor with shape (1,2,3)
my_var = tf.get_variable("my_var",[1,2,3])
# We can initialize the value with an initializer
my_int_var = tf.get_variable("my_int_var",[1,2,3], dtype=tf.int32,
    initializer=tf.zeros_initializer)
# We can initialize the value with a tensor
other_variable = tf.get_variable("other_variable", dtype=tf.int32,
    initializer=tf.constant([23, 42]))

# Il existe des types de collections
# GLOBAL_VARIABLES - var utilisé par plusieurs devices
# TRAINABLE_VARIABLES - var utilisé par Gradient Descent
# LOCAL_VARIABLES - var utilisé par un seul device et pas trainable
my_local = tf.get_variable("my_local", shape=(),
    collections=[tf.GraphKeys.LOCAL_VARIABLES])

my_non_trainable = tf.get_variable("my_non_trainable",shape=(),
    trainable=False)
# on peut aussi créer des collections personalisées
tf.add_to_collection("my_collection_name", my_local)
tf.get_collection("my_collection_name")

# On peut placer les varibales sur un device précis
with tf.device("/device:GPU:1"):
  v = tf.get_variable("v", [1])
# Il est important de placer les variables au bon endroit entre
# les workers servers et les parameters server, pour cela on utilise
# tf.train.replica_device_setter qui place les var sur parameter server
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"], # 2 parameter server
    # 4 worker server
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}

# v in placed in the parameter server 
# by the replica_device_setter
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])

# II - Init Variables
# Sur tensorflow et dans la couche basse (graph + session)
# Il faut initialiser les variables

# session.run(tf.global_variables_initializer())
# Now all variables are initialized.

# Pour connaitre les variables pas encore initialisées
print(session.run(tf.report_uninitialized_variables()))

# Pour éviter les problèmes de dépendances, on peut utiliser
# initialized_value() qui va attendre que la var soit initialisé
# avant d'initialiser une autre variable
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)

# III - Using Variables
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
# On assigne une valeure a une variable
assignment = v.assign_add(1)
tf.global_variables_initializer().run()

sess = tf.Session()
print(sess.run(assignment)) #1.0 
# or assignment.op.run()

# IV - Sharing variables
# Quand on souhaite créer des couches succintes 
# il faut soit créer soit réutiliser des variables
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
# on tente de créer plusieurs conv layers
# on souhaite créer de nouvelles variables pour chaque couche
# seulement ça ne marche pas comme ça

# Pour créer de nouvelles variables on va définir des scopes
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])

# Si on souhaite réutilier les variables, reuse=True
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)

