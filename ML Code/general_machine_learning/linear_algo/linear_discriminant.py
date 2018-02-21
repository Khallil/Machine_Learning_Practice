# -*- coding: utf-8 -*-

# Khallil Doudou
# Linear Discriminant from Scratch

import doudou_tools
import math


# This code works only for a 2 dimension case (only 2 variate)
# # don't know now how lda works with more than that

# get the mean of each class
def get_mean_and_prob_of_each_class(dataset):
    	class_values = [row[-1] for row in dataset]
        unique = set(class_values)
        class_set = list()
        mean = list()
        prob = list()
        for value in unique:
            class_set = list()
            for row in dataset:
                if row[-1] == value:
                    class_set.append(row[0])
            mean.append(doudou_tools.get_mean(class_set))
            prob.append(doudou_tools.get_probability(len(class_set),len(dataset)))
        return mean,prob

def get_squared_difference(dataset,mean):
    #get the squared difference
    class_values = [row[-1] for row in dataset]
    unique = set(class_values)
    sd = list()
    for value in unique:
        sum = 0
        for row in dataset:
            if row[-1] == value:
                sum += ((row[0] - mean[value]) **2)
        sd.append(sum)
    return sd

def get_variance(dataset,sd):
    class_values = [row[-1] for row in dataset]
    unique = set(class_values)
    return(1/(float(len(dataset)) - float(len(unique)) * sum(sd)))

def discriminant_function(mean,prob,variance):
    #xi * mean / variance -(mean**2)/(2*variance)+ math.log(p(class))
    class_values = [row[-1] for row in dataset]
    unique = set(class_values)
    predict_set = list()
    for row in dataset:
        d=[-999,-1]
        for value in unique:
            n = row[0] * mean[value] / variance - (mean[value]**2)/(2*variance)+math.log(prob[0])
            print ("n : ", n,", value : ", value)
            if n > d[0]:
                d[0] = n
                d[1] = value
        predict_set.append(d[1])
    return predict_set
    

dataset = [[4.667797637,0],
[5.509198779,0],
[4.702791608,0],
[5.956706641,0],
[5.738622413,0],
[20.74393514,1],
[21.41752855,1],
[20.57924186,1],
[20.7386947,1],
[19.44605384,1]
]

mean,prob = get_mean_and_prob_of_each_class(dataset)
sd = get_squared_difference(dataset,mean)
variance = get_variance(dataset,sd)
p_set = discriminant_function(mean,prob,variance)
print p_set