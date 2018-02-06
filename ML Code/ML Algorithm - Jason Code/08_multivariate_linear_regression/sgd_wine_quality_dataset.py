
# Linear Regression With Stochastic Gradient Descent for Wine Quality
# with Normalizing Data so we need to get the min max for each column

from random import seed
from random import randrange
from csv import reader
from math import sqrt

#Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r' ) as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Find the min and max values for each column
# retourne une liste de [min,max] pour chaque colonne
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        #get toute les valeur de la colonne i
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
# parcours tout le tableau et normalize les data
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds) # 10 / 3 = 3
    for _ in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Simple linear regression algorithm
def simple_linear_regression(train, test):
    predictions = list()
    coef = coefficients_sgd(train,0.01, 50)
    for row in test:
        yhat = coef[0]
        for i in range(len(coef)-1):
            yhat += coef[i+1] * row[i]
        predictions.append(yhat)
    return predictions

def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse

# Calculate root mean squared error
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    return yhat

def coefficients_sgd(train, l_rate,n_epoch):
    # creer un coeff pour chaque variable de l'equation
    # remplis un tableau de coef par des 0.0
    coef = [0.0 for i in range(len(train[0]))]
    # pour chacune iteration fixe
    for epoch in range(n_epoch):
        sum_error = 0
        # pour chaque row
        for row in train:
            yhat = predict(row,coef)
            error = yhat - row[-1]
            sum_error += error **2

            # update B0
            coef[0] = coef[0] - l_rate * error
            # update B1
            for i in range(len(row)-1): #loop for multivariate
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]

        print ('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef

seed(1)
# load and prepare data
filename = 'winequality-white.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
rmse = evaluate_algorithm(dataset,simple_linear_regression)
print('RMSE : %.3f' % rmse)
# on a maintenant les data normalises
  ## il faut maintenant split les data en train/test
# ensuite on trouve les coefficients avec le gradient descent
  ## ensuite on passe le test set avec les coefficients
# on passe les predictions dans le rmse
# on affiche les scores
