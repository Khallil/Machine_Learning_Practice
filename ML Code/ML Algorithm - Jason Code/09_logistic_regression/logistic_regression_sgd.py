
# Example of Logistic Regression with SGD
# On normalise les data

from random import seed
from random import randrange
from csv import reader
from math import exp

# Load a CSV file
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

# Find the min and max values for each column (needed for normalize data)
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list() # on cree une nouvelle liste
    dataset_copy = list(dataset) # on copy l'original
    fold_size = int(len(dataset) / n_folds) # on calcule la taille de chaque fold
    for _ in range(n_folds): # pour chaque n_fold
        fold = list() # on cree un nouveau fold
        while len(fold) < fold_size: # tant que le fold est pas rempli
            index = randrange(len(dataset_copy)) # on assigne aleatoirement une valeur du dataset
            fold.append(dataset_copy.pop(index)) # on retire la valeur du dataset
            # et on l'ajoute au nouveau fold
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
# on prends le dataset, on enleve un fold, ce fold servira de test_set
# tandis que tous les autres serviront a entrainer le model
# en prenant soin de faire passer tous les fold dans cette procedure

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds) # copie folds en train_set
        train_set.remove(fold) # enleve la fold du train_set
        train_set = sum(train_set, []) # ajoute le train_set dans []
        # ce qui annule la folderization ,[f1,f2,f3,...] 
        test_set = list()
        for row in fold: 
            row_copy = list(row)
            test_set.append(row_copy) # ajoute a test_set 
            row_copy[-1] = None # set a None y
            
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold] #get tous les y du fold
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    # pour chaque x dans la row, sans la derniere colonne (y)
    for i in range(len(row)-1):
        yhat += coefficients[i + 1] * row[i]
    res = 1.0 / (1.0 + exp(-yhat))
    print res
    return res

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        #sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
         #  sum_error += error**2
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row)-1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
        #print( ' >epoch=%d, lrate=%.3f, error=%.3f ' % (epoch, l_rate, sum_error))
    return coef

# Logistic Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

# Test the logistic regression algorithm on the diabetes dataset

# load and prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

seed(1)
n_folds = 5
l_rate = 0.1
n_epoch = 100

scores = evaluate_algorithm(dataset, logistic_regression, n_folds,l_rate,n_epoch)
#scores = [1,2,3
print('Scores : %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
