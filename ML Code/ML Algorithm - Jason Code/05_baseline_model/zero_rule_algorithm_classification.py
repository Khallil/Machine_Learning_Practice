# Example of Zero Rule Classification Predictions

# Se base sur la classe la plus apparente du train set
# et la predit pour chaque unseen data

from random import seed

def zero_rule_algorithm_classification(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

seed(3)
train = [['2'], ['1'],['2'], ['2'], ['3'], ['3']]
test = [[None], [None], [None], [None]]
predictions = zero_rule_algorithm_classification(train, test)
print(predictions)
# ['2', '2', '2', '2']