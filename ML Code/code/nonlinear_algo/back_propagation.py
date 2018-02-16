# -*- coding: utf-8 -*-

#Doudou Khallil
#Back Propagation Algorithm From Scratch - One Hidden Layer - One Neuron

# Example of initializing a network
from random import seed
from random import random
from math import exp
import numpy as np



# Calculate neurone activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neurone activation
def transfer(activation):
    result =  1.0 / (1.0 + exp(-activation))
    return result

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Transfer Derivative
def get_derivate(output):
    derivative = output * (1.0 - output)
    return derivative

# Back propagate through layers
def back_propagate(network, classes_values):
    # 1 -> 0 | Ouput -> Hidden
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        # si ce n'est pas l'Output layer
        if i != len(network)-1:
            # 0 | 1 seul neurone
            for j in range(len(layer)):
                error = 0.0
                # pour les neuron dans l'Output layer | donc 3
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:   
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(classes_values[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]                  neuron['delta'] = errors[j] * get_derivate(neuron['output'])

def update_weights(network,row,alpha):
    # pour chaque layer
    for i in range(len(network)):
        # pour la première couche
        inputs = row[:-1]
        # pour les suvantes
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i -1]]
        for neuron in network[i]:
            # pour chaque inputs, un weight donc ça marche
            for j in range(len(inputs)):
                neuron['weights'][j] += alpha * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += alpha * neuron['delta']

def train_network(network,train,alpha,n_epoch,n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            # on forward pour init les output des neurones
            outputs = forward_propagate(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            # on backpropagate pour init les delta
            back_propagate(network,expected)
            # avec output et delta, on peut mettre a jour les poids
            update_weights(network,row,alpha)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, sum_error))

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights' :[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Test initializing a network
seed(1)
dataset = [
[3.393533211,2.331273381,0],
[3.110073483,1.781539638,0],
[1.343808831,3.368360954,0],
[3.582294042,4.67917911,0],
[2.280362439,2.866990263,0],
[7.423436942,4.696522875,1],
[5.745051997,3.533989803,1],
[9.172168622,2.511101045,1],
[7.792783481,3.424088941,1],
[7.939820817,0.791637231,1]]

testset = [
[3.393533211,2.331273381],
[3.110073483,1.781539638],
[1.343808831,3.368360954],
[3.582294042,4.67917911],
[2.280362439,2.866990263],
[7.423436942,4.696522875],
[5.745051997,3.533989803],
[9.172168622,2.511101045],
[7.792783481,3.424088941],
[7.939820817,0.791637231]]

print dataset
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 4, n_outputs)
train_network(network, dataset,0.1,1000,n_outputs)

for row in testset:
    print predict(network,row)


