# -*- coding: utf-8 -*-

# Khallil Doudou
# Perceptron
# The goal now is to binary classify

# multivariate gradient descent
def multivariate_gradient_descent(dataset, alpha, n_epoch):
    coef_set = [0.0 for i in range(len(dataset[0]))]
    while (n_epoch > 0):
        for x in range(len(dataset)):
            p_y = predict(coef_set,dataset[x])
            error = dataset[x][-1] - p_y
            coef_set[0] = coef_set[0] + alpha * error
            for i in range(len(coef_set)-1):
                coef_set[i+1] = coef_set[i+1] + alpha * error * dataset[x][i]
        n_epoch -= 1
    return coef_set

# predict
def predict(coef_set,row):
    sum = coef_set[0]
    for x in range(len(row)-1):
        sum += coef_set[x+1] * row[x]
    return 1.0 if sum >= 0.0 else 0.0


# predict_set(coef_set,dataset)
def predict_set(coef_set,dataset):
    predicted_set = list()
    for row in dataset:
        predicted_set.append(predict(coef_set,row))
    return predicted_set

# Calculate weights
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

l_rate = 0.1
n_epoch = 
coef_set = multivariate_gradient_descent(dataset,l_rate,n_epoch)
p_set = predict_set(coef_set,dataset)
for item in p_set:
    print item
