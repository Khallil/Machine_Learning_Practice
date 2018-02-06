
#Standardiser vise a connaitre la difference d'une valeur
#par rapport a la moyenne des valeurs, dans une range -1 a 1 pour
# une standard deviation de 1

from math import sqrt

# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means

# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i]-means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
    return stdevs

# standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]

dataset = [[50, 30], [20, 90], [30, 50]]
print(dataset)

means = column_means(dataset)
stdevs = column_stdevs(dataset,means)

print(means)
print(stdevs)

standardize_dataset(dataset, means, stdevs)
print(dataset)