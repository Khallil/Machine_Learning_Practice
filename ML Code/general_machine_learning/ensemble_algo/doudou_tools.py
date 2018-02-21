
# Hehe c ma librari

# STAT PROPERTIES
def get_mean(list_v):
    sum = 0
    for x in list_v:
        sum += x
    mean =  sum/len(list_v)
    return mean

def get_probability(count_c,count_n):
    return float(count_c) / float(count_n)

# TYPE CONVERSION
def convert_int_dataset_to_float_dataset(dataset):
    converted_dataset = list()
    float_list = list()
    for instance in dataset:
        float_list = list()
        for x in instance:
            float_list.append(float(x))
        converted_dataset.append(float_list)
    return converted_dataset
    
def convert_int_to_float(int_list):
    float_list = list()
    for x in int_list:
        float_list.append(float(x))
    return float_list

# ACCURACY
def get_accuracy_of_prediction_classification(y_set,p_set):
    correct = 0
    for i in range(len(p_set)):
        if y_set[i] == p_set[i]:
            correct += 1
    return correct / float(len(y_set)) * 100.0

def get_accuracy_of_prediction_regression(y_set, p_set):
    error = list()
    for i in range(len(y_set)):
        if (y_set[i] == 0):
            error.append(abs(y_set[i] - p_set[i]) * 100)
        else:
            error.append((abs(y_set[i] - p_set[i]) / y_set[i]) * 100)
    return(get_mean(error))

# SCALING DATA
# range(0-1)
def normalize_dataset_without_y(dataset):
    # doit get toutes les valeurs de chaque colonne sans y
    for i in range(len(dataset[0])-1):
        col_values = [row[i] for row in dataset]
        # obtenir le max et le min
        max_ = max(col_values)
        min_ = min(col_values)
        t = 0
        # appliquer la formule de normalisaton a chaque element de la colonne
        for row in dataset:
            row[i] = (row[i] - min_) / (max_ - min_)
            t += 1
    # a la fin de la boucle on retourne le dataset
    return dataset

def normalize_dataset_with_y(dataset):
    # doit get toutes les valeurs de chaque colonne sans y
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        # obtenir le max et le min
        max_ = max(col_values)
        min_ = min(col_values)
        t = 0
        # appliquer la formule de normalisaton a chaque element de la colonne        
        for row in dataset:
            row[i] = (row[i] - min_) / (max_ - min_)
            t += 1
    # a la fin de la boucle on retourne le dataset
    return dataset

'''dataset = [[1,3,4,5,1],
            [1,6,5,2,0], 
            [1,5,3,2,1],
            [2,3,5,1,0],
            [7,6,9,1,1]]
'''