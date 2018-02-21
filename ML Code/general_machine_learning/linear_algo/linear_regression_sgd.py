# -*- coding: utf-8 -*-

# Ce code est coder pour gerer les cas ou dataset = [[X1],[X2],[XN],[Y]]
# alors que communement le dataset = [[X1,X2,XN,Y],[X1,X2,XN,Y]]

# Khallil Doudou
# Linear Regression with Stochastic Gradient Descent from scratch

import doudou_tools

def gradient_descent_process(n_loop, alpha,training_set):
    coef = [0.0 for i in range(len(training_set))]
    while n_loop > 0:
        i = 0
        for row in range(len(training_set[0])):
            sum = coef[0]
            for x in range(len(training_set)-1):
                sum += coef[x+1] * training_set[x][i]           
            error = sum - training_set[-1][i]        
            coef[0] = coef[0] - alpha * error
            for x in range(len(training_set)-1):
                print "coef[x=",x+1, coef[x+1]
                coef[x+1] = coef[x+1] - alpha * error * training_set[x][i]
            i += 1
            print " "
        n_loop -= 1
    return coef


def predict(testing_set,coef):
    predict_set = list()
    print(coef)
    i = 0
    for row in range(len(testing_set[0])):
        p = coef[0]
        for x in range(len(testing_set)-1):
            p += coef[x+1] * testing_set[x][i]
        print(p)
        predict_set.append(p)
        i += 1
    return predict_set


# on part sur un vrai dataset
x_set = dataset = [[1,3,4,5,6],[1,6,5,2,0], [1,5,3,2,1]]

#x_set = doudou_tools.convert_int_to_float(x_set)
converted_dataset = doudou_tools.convert_int_dataset_to_float_dataset(x_set)
print(converted_dataset)
#y_set = doudou_tools.convert_int_to_float(y_set)
n_loop = 50
alpha = 0.01

coef = gradient_descent_process(n_loop,alpha,converted_dataset)
print "coef : ", coef
predict_set = predict(converted_dataset,coef)
#print "p_set : ", predict_set
print(doudou_tools.get_accuracy_of_prediction_regression(converted_dataset[-1],predict_set))