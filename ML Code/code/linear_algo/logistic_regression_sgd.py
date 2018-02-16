# -*- coding: utf-8 -*-

# Khallil Doudou
# Logistic Regression from scratch
# The goal now is to classify

import doudou_tools
from math import exp

# set to float ok
# normalize data

# gradient descent function
def multivariate_gradient_descent(dataset, alpha, n_epoch):
    coef_set = [0.0 for i in range(len(dataset[0]))]
    while (n_epoch > 0):
        for x in range(len(dataset)):
            #print(coef_set)
            p_y = predict(coef_set,dataset[x])
            error = dataset[x][-1] - p_y
            coef_set[0] = coef_set[0] + alpha * error * p_y * (1 - p_y)
            for i in range(len(coef_set)-1):
                coef_set[i+1] = coef_set[i+1] + alpha * error * p_y * (1 - p_y) * dataset[x][i]
        n_epoch -= 1
        print coef_set    
    return coef_set

# predict function (row,y)
def predict(coef_set,row):
    sum = coef_set[0]
    for x in range(len(row)-1):
        sum += coef_set[x+1] * row[x]
    res = 1.0 / (1.0 + exp(-sum))
    print res
    return res

# predict_set(coef_set,dataset)
def predict_set(coef_set,dataset):
    predicted_set = list()
    for row in dataset:
        predicted_set.append(round(predict(coef_set,row)))
    return predicted_set

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

dataset = doudou_tools.convert_int_dataset_to_float_dataset(dataset)
dataset = doudou_tools.normalize_dataset_with_y(dataset)

for item in dataset:
    print item
alpha = 0.2
n_epoch = 10
coef_set = multivariate_gradient_descent(dataset,alpha,n_epoch)
predict_set = predict_set(coef_set,dataset)
y_set = [row[-1] for row in dataset]
print(predict_set)
print(doudou_tools.get_accuracy_of_prediction_classification(y_set,predict_set))