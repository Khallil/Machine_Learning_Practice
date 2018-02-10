#-*- coding: utf-8 -*-

# Doudou Khallil
# Stacked Generalization from scratch

import doudou_tools
from math import exp
# On a nos prédictions provenant des autres algorithmes
y_knn = [1,0,0,0,1]
y_perceptron = [0,0,1,1,1]
# On a les y originaux
y_dataset = [0,1,0,1,0]



# On crée un nouveau dataset avec les y_set
# 0 0 0
# 1 0 1

def get_new_dataset_with_y(y_set):
    dataset = list()
    for x in range(len(y_set[0])):
        y_new = list()
        for i in range(len(y_set)):
            y_new.append(y_set[i][x])
        dataset.append(y_new)
    return dataset

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
    return coef_set

# predict function (row,y)
def predict(coef_set,row):
    sum = coef_set[0]
    for x in range(len(row)-1):
        sum += coef_set[x+1] * row[x]
    res = 1.0 / (1.0 + exp(-sum))
    return res

# predict_set(coef_set,dataset)
def predict_set(coef_set,dataset):
    predicted_set = list()
    for row in dataset:
        predicted_set.append(round(predict(coef_set,row)))
    return predicted_set

y_set = list()
y_set.append(y_knn)
y_set.append(y_perceptron)
y_set.append(y_dataset)

dataset = get_new_dataset_with_y(y_set)

for item in dataset:
    print item

coef = multivariate_gradient_descent(dataset,0.3,1000)
y_final = predict_set(coef,dataset)

for item in y_final:
    print item

print ("KNN result : ",doudou_tools.get_accuracy_of_prediction_classification(y_dataset,y_knn))
print ("Perceptron result : ",doudou_tools.get_accuracy_of_prediction_classification(y_dataset,y_perceptron))
print ("Stacked with Logistic Regression result : ",doudou_tools.get_accuracy_of_prediction_classification(y_dataset,y_final))
