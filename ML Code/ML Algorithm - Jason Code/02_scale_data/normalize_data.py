
#Normaliser vise a rescale les valeurs dans une range de 0 a 1
#pour 0 le minimum et 1 le maximum des valeurs

def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]    
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax

# rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
            #50 = (50 - 20) / 50 - 20 = 30 / 30 = 1
            #30 = (30 - 20) / 50 - 20 = 10 / 30 = 0.33

# Contrive small dataset
dataset = [[50.0, 30.0], [30.0, 50.0], [20.0, 90.0]]

minmax = dataset_minmax(dataset)
print(minmax)

print(dataset)
# Calculate min and max for each column
normalize_dataset(dataset,minmax)

print(dataset)
