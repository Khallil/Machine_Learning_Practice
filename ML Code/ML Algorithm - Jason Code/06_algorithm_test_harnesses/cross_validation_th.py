# Example of a Cross Validation Test Harness

from random import seed
from random import randrange
from csv import reader

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

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Algorithm Evaluation : calculate accuracy metric
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100

# Split_Data : cross_validation split (k_fold)
# + run l'accuracy_metric
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Naive Baseline Model : Zero Rule Algorithm for classification (predit le plus frequent)
def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

# Test the train/test harness

#set le random
seed(1)

#load an prepare data
filename = 'pima-indians-diabetes.csv'
dataset = load_csv(filename)

#transform the string in dataset in float value
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

#nb of folds
n_folds = 5

algorithm = zero_rule_algorithm_classification
algorithm_name = "zero_rule_algorithm_classification"
scores = evaluate_algorithm(dataset,algorithm,n_folds)

print('Dataset : %s ' % filename)
print('Split method : cross_validation - nb_fold = %.1f ' % (n_folds))
print('Scale data method : %s ' % "none")
print('Algorithm : %s ' % algorithm_name)
print('Scores: %s ' % scores)
print('Accuracy : %.3f%%' % (sum(scores)/len(scores)))
