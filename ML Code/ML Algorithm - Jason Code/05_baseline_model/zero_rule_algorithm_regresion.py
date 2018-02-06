# Example of Zero Rule Regression Predictions

# It will calculate the mean, then it will only predict
# the mean value for each x of unseen data

from random import seed
# zero rule algorithm for regression
def zero_rule_algorithm_regression(train, test):
    output_values = [row[-1] for row in train]
    prediction = sum(output_values) / float(len(output_values))
    predicted = [prediction for i in range(len(test))]
    return predicted

seed(1)
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_algorithm_regression(train, test)
print(predictions)
#[15.0, 15.0, 15.0, 15.0]