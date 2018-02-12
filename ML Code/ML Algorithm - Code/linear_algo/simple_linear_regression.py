# -*- coding: utf-8 -*-

# Khallil Doudou
# Simple linear regression for R2 from scratch

# The goal is to find the B0 and B1 coefficients
# We will use the simple linear regression technique
# B1 = E(xi - mean(x)) * (yi - mean(y)) / E(xi – mean(x))^2
# then B0 = mean(y) - B1 * mean(x)

import doudou_tools

# LEARNING 
# 1. return xi-mean(x)
def get_x_minus_mean_of_x(x_set, mean_x):
    x_minus_mean_of_x = list()
    for x in x_set:
        x_minus_mean_of_x.append(x - mean_x)
    return x_minus_mean_of_x

# 2. return yi-mean(x)
def get_y_minus_mean_of_y(y_set, mean_y):
    y_minus_mean_of_y = list()
    for y in y_set:
        y_minus_mean_of_y.append(y - mean_y)
    return y_minus_mean_of_y

# 2. return  E(xi - mean(x)) * (yi - mean(y))
def get_nominator(x_set,y_set,mean_x, mean_y):
    x_minus_mean_of_x = get_x_minus_mean_of_x(x_set,mean_x)
    y_minus_mean_of_y = get_y_minus_mean_of_y(y_set,mean_y)
    sum = 0
    for x,y in zip(x_minus_mean_of_x, y_minus_mean_of_y):
        sum +=  x * y
    return sum

# 3. return E(xi - mean(x)^2)
def get_denominator(x_set,mean_x):
    x_minus_mean_of_x = get_x_minus_mean_of_x(x_set,mean_x)
    sum = 0
    for x in x_minus_mean_of_x:
        sum += x ** 2
    return sum

# 4. return B1 : E(xi - mean(x)) * (yi - mean(y)) / E(xi – mean(x))^2
def get_B1(x_set,y_set,mean_x,mean_y):
    return (get_nominator(x_set,y_set,mean_x,mean_y)/get_denominator(x_set,mean_x))

#5. return B0 : mean(y) - B1 * mean(x)
def get_B0(x_set,y_set,mean_x,mean_y):
     return(mean_y - get_B1(x_set,y_set,mean_x,mean_y) * mean_x)

# PREDICT
def get_predict_value(x_set,y_set,mean_x,mean_y,new_x_set):
    predicted_set = list()
    for new_x in new_x_set:
        predicted_set.append(get_B0(x_set,y_set,mean_x,mean_y) + get_B1(x_set,y_set,mean_x,mean_y) * new_x)
    return predicted_set

#Dataset
x_set = [1,2,4,3,5]
y_set = [1,3,3,2,5]

x_set = doudou_tools.convert_int_to_float(x_set)
y_set = doudou_tools.convert_int_to_float(y_set)

mean_x = doudou_tools.get_mean(x_set)
mean_y = doudou_tools.get_mean(y_set)

new_x_set = [1,2,4,3,5]
new_x_set = doudou_tools.convert_int_to_float(new_x_set)

predicted_set = get_predict_value(x_set,y_set,mean_x,mean_y,new_x_set)
print "y_set : ", y_set
print "p_set : ", predicted_set
print doudou_tools.get_accuracy_of_prediction_regression(y_set,predicted_set)