
# Example of Calculating Mean Absolute Error

# permet de connaitre l'ecart moyen d'erreur pour chaque prediction
# cette mesure est utile en cas d'algorithme de regression

# Calculate mean absolute error
def mae_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))

# Test RMSE
actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print(mae)
#0.008
#ce qui veut dire que l'ecart moyen d'erreur est  0.008
#ce qui est inferieur a l'ecart moyen attendu de 0.01 donc good
