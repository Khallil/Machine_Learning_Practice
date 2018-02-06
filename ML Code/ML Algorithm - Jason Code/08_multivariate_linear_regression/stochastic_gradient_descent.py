#Example of a simple stochastic_gradient program

# the specific of sgd, is that for each row we update the coefficients
# all row are used for update for each fixed number of epoch

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def coefficients_sgd(train, l_rate,n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    print("coef : ",coef)
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row,coef)
            error = yhat - row[-1]
            sum_error += error **2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row)-1): 
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print ('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef

dataset = [[1,3,4,5,6],[1,6,5,2,0], [1,5,3,2,1]]
l_rate = 0.01
n_epoch = 10
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print coef
